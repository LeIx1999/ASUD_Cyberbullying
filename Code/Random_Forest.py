import numpy as np
from sklearn.preprocessing import LabelEncoder

# Encode the classes
def encode_classes(classes: list) -> np.array():
    encoder = LabelEncoder()
    classes_trans = encoder.fit_transform(classes)
    return classes_trans, encoder

# decode classes
def decode_classes(preds: list, encoder: LabelEncoder) -> list:
    encoder_dict = dict(enumerate(encoder.classes_.flatten(), 0))
    preds = [encoder_dict[x] for x in preds]
    return preds

# calculate accuracy
def calc_accuracy(preds: list, act_values: list) -> str:
    acc = 0
    for i in range(len(preds)):
        if preds[i] == act_values[i]:
            acc += 1
    accuracy = acc / len(preds)
    return f'Accuracy: {str(accuracy)}'

# compare value counts of predictions and acutal values
def compare_value_counts(preds: list, act_values: list) -> list:
    vc_preds = [[x, preds.count(x)] for x in act_values.unique().to_list()]
    vc_act = preds.value_counts()
    return [vc_preds, vc_act]