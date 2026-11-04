# Nightfall — adaptive enemy AI (TFLite prototype)

This prototype is based on the *Nightfall* presentation: the enemy adapts to player hiding, running, flashlight usage, health, and time spent in darkness.

## Create the `.tflite` model

In **Google Colab**, upload `train_nightfall.py`, then run:

```python
!pip -q install tensorflow numpy
!python train_nightfall.py
```

Download `nightfall_adaptive_ai.tflite` and `nightfall_model_info.json` from the Colab file browser. Alternatively, run the same commands locally after installing Python and TensorFlow.

## Model contract

- Input: float32 `[1,5]` in this exact order: hiding frequency, running fraction, flashlight-on fraction, health fraction, darkness fraction. Each must be normalized to 0–1.
- Output: float32 `[1,4]`, probabilities in this exact order: `CHECK_HIDING_SPOTS`, `AGGRESSIVE_CHASE`, `DARKNESS_SCARE`, `AMBUSH_SAFE_ROUTE`.
- Take `argmax` to select the most likely tactic. Use a cooldown and gradual transition; do not immediately counter every player action.
- Training labels are **synthetic, rule-derived examples**. Test accuracy measures how well the model imitates those rules, **not** how accurately it predicts human emotion.

## Example

Input `[0.82, 0.46, 0.74, 0.70, 0.22]` represents a player who hides often, runs sometimes, relies on the flashlight, has 70% health and spends 22% of the time in darkness. The model predicts a tactic based on these features.

## Unity integration note

Unity ML-Agents normally exports **ONNX**, not TFLite. A TFLite model needs a compatible Unity TensorFlow Lite runtime/plugin or an external inference service; it cannot be directly dropped into ML-Agents as an ONNX policy. The TFLite output should drive a separate behaviour-tree or state-machine controller. Keep escape routes, cooldown periods, and intensity settings enforced by game logic.
