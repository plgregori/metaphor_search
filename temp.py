xml_true = "The market was a <Metaphor> roller coaster </Metaphor> yesterday , and many called it a <Metaphor> bloodbath </Metaphor> ."
xml_pred = "The market was a <Metaphor> roller </Metaphor> coaster yesterday , and many called it a <Metaphor> total bloodbath </Metaphor> ."
print("xml_true:\n", xml_true)
print("\nxml_pred:\n", xml_pred)

from src.evaluation import do_praf
results = do_praf(xml_true, xml_pred, "Metaphor")

print(results.precision)
print(results.recall)
print(results.accuracy)
print(results.f1)
print(results.confusion_matrix)