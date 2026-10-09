from pandas import DataFrame, isna, read_csv, to_numeric
from pandas.errors import EmptyDataError, ParserError
from src.errors import PromptFileError
from typing import get_args
from src.prompt_engineering.schemas import PromptMessage, PromptStrategy, Role
import logging

logger = logging.getLogger("uvicorn.error")

PLACEHOLDER = "[#TEST_TEXT]"
_VALID_ROLES = set(get_args(Role))
_REQUIRED_COLUMNS = ("pid", "name", "cid", "role", "content")


def _csv_line(index: int) -> int:
    return index + 2                                       # header is line 1, the first data row is line 2

def _check_prompt_table(prompts_df: DataFrame) -> list[str]:
    """Returns every problem found (empty list = valid), so you can fix a hand-edited file in one go."""
    problems: list[str] = []

    for column in ("pid", "cid"):
        numeric = to_numeric(prompts_df[column], errors="coerce")
        for index in prompts_df.index[numeric.isna() | (numeric % 1 != 0)][:5]:
            value = prompts_df.at[index, column]
            problems.append(f"CSV line {_csv_line(index)}: '{column}' must be a whole number, got {'nothing' if isna(value) else repr(value)}")
    for column in ("name", "role", "content"):
        blank = prompts_df[column].isna() | (prompts_df[column].astype(str).str.strip() == "")
        for index in prompts_df.index[blank][:5]:
            problems.append(f"CSV line {_csv_line(index)}: '{column}' is empty")
    if problems:
        return problems                                    # the per-strategy checks below need clean pid/cid/role/content

    table = prompts_df.assign(pid=to_numeric(prompts_df["pid"]).astype(int), cid=to_numeric(prompts_df["cid"]).astype(int))
    for pid, group in table.groupby("pid"):
        group = group.sort_values("cid")
        label = f"Strategy {pid}"

        if group["name"].nunique() != 1:
            problems.append(f"{label}: its rows have different names {sorted(set(group['name']))}")

        cids = group["cid"].tolist()
        if cids != list(range(len(cids))):
            problems.append(f"{label}: cid must run 0, 1, 2, ... without gaps or repeats, got {cids}")

        roles = group["role"].tolist()
        unknown_roles = sorted(set(roles) - _VALID_ROLES)
        if unknown_roles:
            problems.append(f"{label}: unknown role(s) {unknown_roles}; allowed: {sorted(_VALID_ROLES)}")
        else:
            turns = roles[1:] if roles[0] == "system" else roles
            expected = ["user" if i % 2 == 0 else "assistant" for i in range(len(turns))]
            if not turns or turns != expected or turns[-1] != "user":
                problems.append(f"{label}: expected an optional system message, then alternating user/assistant messages "
                                f"ending with a user message; got {roles}")

        contents = group["content"].astype(str).tolist()
        occurrences = sum(content.count(PLACEHOLDER) for content in contents)
        if occurrences != 1 or PLACEHOLDER not in contents[-1]:
            problems.append(f"{label}: {PLACEHOLDER} must appear exactly once, in the last message; "
                            f"found {occurrences} time(s), {'in' if PLACEHOLDER in contents[-1] else 'not in'} the last message")
    return problems

class PENode:
    prompts_path: str
    strategies: dict[int, PromptStrategy]

    def __init__(self, prompts_path: str) -> None: 
        self.prompts_path = prompts_path
        try:
            prompts_df: DataFrame = read_csv(prompts_path)
        except FileNotFoundError as e:
            raise PromptFileError(f"Prompts file not found: {prompts_path}") from e
        except (OSError, UnicodeDecodeError, EmptyDataError, ParserError) as e:
            raise PromptFileError(f"Prompts file could not be read ({prompts_path}): {e}") from e

        missing_columns = set(_REQUIRED_COLUMNS) - set(prompts_df.columns)
        if missing_columns:
            raise PromptFileError(f"Prompts file is missing required columns: {sorted(missing_columns)}")
        if prompts_df.empty:
            raise PromptFileError(f"Prompts file {prompts_path} has no rows.")
        problems = _check_prompt_table(prompts_df)
        if problems:
            raise PromptFileError(f"Prompts file {prompts_path} has {len(problems)} problem(s):\n- " + "\n- ".join(problems))

        prompts_df = prompts_df.assign(pid=to_numeric(prompts_df["pid"]).astype(int),
                                       cid=to_numeric(prompts_df["cid"]).astype(int))

        self.strategies = {}
        for _, group in prompts_df.sort_values(["pid", "cid"]).groupby("pid", sort=False):
            pid = int(group["pid"].iloc[0])                # the group key is typed as a generic pandas Scalar; the column value is not
            self.strategies[pid] = PromptStrategy(
                pid=pid,
                name=str(group["name"].iloc[0]),
                messages=tuple(PromptMessage(role=role, content=content)
                               for role, content in zip(group["role"], group["content"])),
            )

    def build_messages(self, pid: int, plain_text: str) -> list[dict[str, str]]:
        """The strategy's messages with the test text in place of the placeholder, as plain dicts for the provider SDKs."""
        return [{"role": m.role, "content": m.content.replace(PLACEHOLDER, plain_text)}
                for m in self.strategies[pid].messages]