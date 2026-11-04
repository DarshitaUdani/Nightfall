"""Train and export Nightfall's behaviour-adaptive TFLite classifier.
Run: pip install tensorflow numpy && python train_nightfall.py
All training samples are synthetic and must not be represented as real playtest data.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

OUT = Path(__file__).resolve().parent
SEED = 42
rng = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)
FEATURES = ['hiding_frequency', 'running_fraction', 'flashlight_fraction', 'health_fraction', 'darkness_fraction']
ACTIONS = ['CHECK_HIDING_SPOTS', 'AGGRESSIVE_CHASE', 'DARKNESS_SCARE', 'AMBUSH_SAFE_ROUTE']

# Normalized features are all fractions in [0,1]. The scores are deliberately
# authored gameplay heuristics, NOT measured fear or validated player labels.
def policy_scores(x):
    h, r, f, hp, d = x.T
    return np.stack([
        1.8*h + .15*(1-r) + .10*(1-d),
        1.8*r + .15*hp + .05*(1-h),
        1.8*f + .18*(1-d) + .05*(1-r),
        1.0*(1-h) + .55*(1-r) + .35*d + .10*hp,
    ], axis=-1)

X = rng.uniform(0, 1, size=(12000, len(FEATURES))).astype('float32')
# Slight random variation makes the synthetic classification task less brittle.
noisy_scores = policy_scores(X) + rng.normal(0, .06, size=(len(X), len(ACTIONS)))
y = np.argmax(noisy_scores, axis=1).astype('int32')
idx = rng.permutation(len(X))
train, test = idx[:9600], idx[9600:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,), name='player_telemetry'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax', name='tactic_probabilities'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.002),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X[train], y[train], epochs=25, batch_size=128,
          validation_split=.15, verbose=2)
loss, accuracy = model.evaluate(X[test], y[test], verbose=0)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
(OUT / 'nightfall_adaptive_ai.tflite').write_bytes(tflite)
metadata = {
    'inputs': FEATURES, 'input_shape': [1,5], 'input_dtype': 'float32',
    'normalization': 'each input must be a fraction between 0 and 1',
    'outputs': ACTIONS, 'output_shape': [1,4],
    'synthetic_test_accuracy': round(float(accuracy), 4),
    'training_data': 'synthetic heuristic-labelled examples; not real gameplay',
    'integration': 'apply cooldown, gradual adaptation, escape-route, and player intensity constraints in game logic'
}
(OUT / 'nightfall_model_info.json').write_text(json.dumps(metadata, indent=2))

# Validate the actual exported artifact, not only the original Keras network.
interpreter = tf.lite.Interpreter(model_path=str(OUT / 'nightfall_adaptive_ai.tflite'))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
example = np.array([[.82, .46, .74, .70, .22]], dtype=np.float32)
interpreter.set_tensor(inp['index'], example)
interpreter.invoke()
probs = interpreter.get_tensor(out['index'])[0]
print('TFLite output:', dict(zip(ACTIONS, np.round(probs, 3).tolist())))
print('Synthetic holdout accuracy:', accuracy)
print('Saved:', OUT / 'nightfall_adaptive_ai.tflite')
