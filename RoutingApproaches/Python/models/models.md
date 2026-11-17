Here lie all of the models trained.

Versions are split by architecture

Diagrams were produced by Claude Code

# v1
## Architecture
```
     text (B, 5, 32) int32                     features (B, 4) float32
             │                                           │
             ├──────────────┐                            │
             ▼              │                            │
┌──────────────────────────┐│                            │
│ Embedding                ││                            │
│ vocab_size+1 → 64        ││                            │
│ (B, 5, 32, 64)           ││                            │
└────────────┬─────────────┘│ padding mask               │
             ▼              │ (token ≠ 0)                │
┌──────────────────────────┐│                            │
│ MessagePooling           │◄┘                           │
│ masked mean over tokens  │                             │
│ (B, 5, 64)               │                             │
└────────────┬─────────────┘                             │
             ▼                                           ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│ GlobalAveragePooling1D   │              │ Dense(16, relu)          │
│ mean over messages       │              │ (B, 16)                  │
│ (B, 64)                  │              │                          │
└────────────┬─────────────┘              └────────────┬─────────────┘
             │                                         │
             └────────────────────┬────────────────────┘
                                  ▼
                ┌──────────────────────────┐
                │ Concatenate              │
                │ (B, 80)                  │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Dense(64, relu)          │
                │ (B, 64)                  │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Dense(1, sigmoid)        │
                │ P(person responds)       │
                └────────────┬─────────────┘
                             ▼
                   ≥ ROUTING_CUTOFF (0.5) ?
```

## Models

### adam
Used all of the defaults in commit c7f71bc6ae8cc6885714d10210a9d641b3fd899a. 

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8525
precision  0.3905
recall     0.1848
f1         0.2508
auc        0.5984
```
### eve
Same hyperparameters as above. Despite that, and supposedly having a fixed seed, it performs differently
than adam, so that's strange. Will need to figure out what that is about.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.7956
precision  0.2458
recall     0.2559
f1         0.2508
auc        0.5760
```

### cain
This time with 100 epochs! Looking at the training curve, it mostly plataeus around 30 epochs, with both the AUC and
F1 score almost being perfect by the end. This suggests the model's strugling to generalize (based on the data below) with
this current architecture, so I'll change it for the next model.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.7023
precision  0.2151
recall     0.4632
f1         0.2937
auc        0.6121
```

### abel
Okay I said I'd change the architecture but before that I wanted to try with a low number of epochs, so 5 this time, and this time it's better! So it's probably
overfitting.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8785
precision  0.6308
recall     0.2197
f1         0.3259
auc        0.7269
```

Oops, delete this model accidently. Retrain on old commit with `EPOCHS=5` if you care for it.

# v2
## Architecture

## Models
### stinky
Trained with 5 epochs, default of the commit I am about to push. Based on its F1 score it appears to be doing quite bad, but also better than previous models! 
So that's something. I believe it might do better just wtih more epochs, so I am going to try that. I also added validation scoring onto the history graphs, so
it will be easier to tell what is happening with future models.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8815
precision  0.5958
recall     0.3533
f1         0.4436
auc        0.7904
```

### stinkier
This one I ran with 20 epochs, and the graph was genuinely shocking. It just gets worse and worse after each epoch. By the end, val_loss was like 70%. Horrific.

Definite evidence of overfitting. I will lobotomize the model next.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8557
precision  0.4422
recall     0.3059
f1         0.3616
auc        0.6781
```

### butter
This one had its `TEXT_VECTOR_SIZE` reduced to `8`, which did, at least for the first few epochs, appear to reduce overfitting. `EPOCH=20` still. For the next
model, I will decrease its number of neurons more, and see if that helps.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8752
precision  0.5805
recall     0.2385
f1         0.3381
auc        0.7017
```

Accidently deleted the weights for this one to

### butter 2
Changed inner layer sizes to `4`.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8735
precision  0.5606
recall     0.2484
f1         0.3443
auc        0.7293
```

### margerine
Changed first inner layer size to `8`, and feature inner layer to `8`.

At least until about the 3rd or 4th epoch, the val loss did go down, which is interesting. I do think the previous models were too complicated and just overfitting,
but the val f1 score continuously went down, so I believe I need to reprioritize the metrics.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8688
precision  0.5424
recall     0.1199
f1         0.1963
auc        0.6097
```

### quickshot
Added early stopping based on the f1_val_score metric. It does perform better, but the f1 score ends up being just
slightly above 0.5, so not much better than just guessing no each time.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8874
precision  0.5910
recall     0.5106
f1         0.5479
auc        0.7268
```

### chicken pot pie
Started weighing positives more based on what percentage of the validation set was positive. So 9x more. This worked GREAT! The training metrics look worse, but
they are pairing with the validation metrics, so it seems we are no longer horrendously overfitting. On top of that, this stopped early at about 5 epochs, and in
that time the f1 score did not decrease. So that is great.

An interesting note about this model, is I tried evaluating it with different routing cutoffs, i.e. thresholds, and
any cutoff >=.5 (increments of 0.1) gave the same evaluation. Ones below got worse, especially <=0.3, which all
also had the same evaluation. I'm guessing that almost all outputs come out around either 0 or 0.5.

Quickshot and chicken pot pie seem to perform almost the same, however.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8874
precision  0.5910
recall     0.5106
f1         0.5479
auc        0.7278
```

### frog
Same as above but removed the early stopping, so it ran on the full 20 epochs. It did go down, but interestingly near the end the val scores started to go up
again. Due to that, I'm going to next try running this with 100 epochs, and see what happens.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.8223
precision  0.3650
recall     0.4457
f1         0.4013
auc        0.7027
```

### climber
Same as above but with 100 epochs. After around 18 epochs, the val loss went up and up and up. All the way to 140% at the end! This is now achievable due to the
class weight (training and validation loss are not comparable.) So it seems that more time to train was not what it needed.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     5993
positives  801
accuracy   0.7255
precision  0.2608
recall     0.5743
f1         0.3587
auc        0.7085
```

# v3
Changed to use  a different learning rate, got usernames for training from a larger
pool than the history window (which means the results here aren't directly comparable to
those from before.)

## Models

### a

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     15908
positives  1159
accuracy   0.9165
precision  0.4272
recall     0.4280
f1         0.4276
auc        0.6831
```

### b
History size -> 20

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     15908
positives  1159
accuracy   0.9004
precision  0.3801
recall     0.5824
f1         0.4600
auc        0.8155
```

### c
Candidate pool -> 100

Values here aren't directly comparable to other models since there is a different number of points

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     43020
positives  1257
accuracy   0.9269
precision  0.2402
recall     0.6945
f1         0.3569
auc        0.8679
```

### d
Candidate pool -> 20
Tokens per message -> 0

Appears that the text input does do something! Will run another test with it completely removed from architecture, though.

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     15908
positives  1159
accuracy   0.8616
precision  0.3024
recall     0.6885
f1         0.4202
auc        0.8509
```

### e
No text input layers

#### Performance
```
=== Evaluating at cutoff 0.5 ===
points     15908
positives  1159
accuracy   0.8784
precision  0.3323
recall     0.6635
f1         0.4428
auc        0.8519
```

# v5
## Architecture
Same layers as v4, but the features input grows from 37 to 66 (`FEATURE_COUNT`). The 29 new features:
- bots: is the speaker / the person a bot (`ubottu` mentions people, but they answer whoever triggered it)
- bursts: how many lines the speaker has aimed at this person since they last spoke, and whether the speaker's last line at them is still unanswered
- continuation: how often the speaker splits a thought over several lines, how long their current run is, whether the line looks unfinished (`...`, `,`, `and`) or finished (`.?!`). Replies get annotated against the *last* line of a burst, so a line that will be continued rarely gets the reply
- the message: acknowledgement ("ok then", "thanks"), ends with `?`, has a question word, opens with this person's name, mentions only this person
- the person: recency rank among the candidates, how often they answered when addressed earlier, whether they were last talking to someone else, whether they have been asking things
- timing: minutes since the person, the speaker and anyone last spoke (needs the new `Message.minute`, parsed from `[HH:MM]`)

Training changes that apply to every v5 model:
- `EarlyStopping` was `monitor="val_precision", mode="min"`, so it restored the epoch with the *worst* precision. Now `monitor="val_auc", mode="max"`
- `POSITIVE_WEIGHT` 3.0 -> 1.0, so the output is a calibrated probability and the cutoff sets the precision/recall trade
- `ROUTING_CUTOFF` 0.5 -> 0.65, picked on dev. Precision is above 0.71 for every cutoff from 0.63 to 0.68, so it isn't a lucky threshold, but it was tuned on dev, so the test split is the honest number

Most remaining false positives look like real replies that the annotators linked to a neighbouring line of the speaker's.

## Models
### burst
The new features and training changes above, everything else as in v4. Early-stopped after 8 epochs, best at epoch 3.

At the old setup's cutoff (0.5) it gets precision 0.640, recall 0.509 (v4 `blgh`: 0.542, 0.633).

#### Performance
```
=== Evaluating at cutoff 0.65 ===
points     15908
positives  1159
accuracy   0.9376
precision  0.7114
recall     0.2425
f1         0.3616
auc        0.9203
```

### oov
`burst` plus `VOCAB_LIMIT = 10_000`: only the 10,000 most common words get their own embedding, the rest become `<OOV>`. The text branch overfits more slowly (val AUC peaks at epoch 4 instead of 2). The `xx` tags are given a huge count so the limit doesn't cut them.

Dev precision/recall by cutoff:
```
cutoff  0.50   0.55   0.60   0.63   0.65   0.70
prec    0.642  0.662  0.685  0.715  0.727  0.789
recall  0.494  0.449  0.368  0.309  0.265  0.168
```

#### Performance
```
=== Evaluating at cutoff 0.65 ===
points     15908
positives  1159
accuracy   0.9392
precision  0.7275
recall     0.2649
f1         0.3884
auc        0.9221
```

### horizon
`oov` plus `REPLY_HORIZON = 20`: messages within 20 lines of the end of their log are left out of training *and* evaluation. The logs stop mid-conversation, so those messages can't show their reply (replies land a median 4 lines later, 95% within 21), and they were all counted as "nobody replied": near the end of a log the reply rate falls from ~0.5 to ~0.2. Dev goes from 15908 to 14590 points.

`evaluate_symbolic.py` and `eval.py` build their points with `util.iter_points`, which doesn't apply the horizon yet, so their numbers aren't directly comparable with this and later models.

On the horizon point set, `oov` gets precision 0.694 at recall 0.4, this one 0.682. The ~0.01 gap is just training noise between seeds, which is what led to `ensemble`.

### ensemble
`horizon` with `ENSEMBLE_SIZE = 5`: five members trained one after another (each with its own early stop), averaged. Didn't help on its own: precision 0.675 at recall 0.4. Precision sits at ~0.67-0.68 for every recall from 0.40 to 0.48, so the features had hit their limit.

To load `burst`, `oov` or `horizon` with today's code, set `FEATURE_COUNT = 66` and `ENSEMBLE_SIZE = 1`. For `ensemble`, `FEATURE_COUNT = 66`.

# v6
## Architecture
v5's ensemble of 5, with the features input growing from 66 to 88 (`compute_turn_features`).

Looking at `ensemble`'s false positives at recall 0.4 (cutoff 0.55, 216 of them):
- 88: the person did answer the speaker, but the annotators linked the answer to a *later* line of the speaker's
- 64: the person spoke, but to other people
- 54: the person said nothing for 30 lines
- 10: the person answered an earlier line of the speaker's

So the 22 new features are about how fast the speaker is going (their usual turn length, their share of the last 5/10/20 lines, lines in the same minute, how busy the channel is, short lines, lines opening with "also"/"btw"/"and") and who the person is paying attention to (how many others they're talking to, whether that conversation is still live, their share of lines aimed at the speaker vs. others, whether someone else is already answering the speaker).

## Models
### turns
`ROUTING_CUTOFF = 0.55`, picked on dev as the cutoff with precision >= 0.7 at recall >= 0.4. Both are only ~0.003 past the mark, so it's tight, and the test split is the honest check.

Dev precision/recall by cutoff:
```
cutoff  0.50   0.52   0.54   0.55   0.56   0.58   0.60
prec    0.675  0.681  0.691  0.703  0.710  0.718  0.729
recall  0.462  0.439  0.414  0.403  0.385  0.354  0.322
```

On every dev point (`REPLY_HORIZON = 0`, 15908 points) it gets precision 0.6909, recall 0.4012, AUC 0.9246.

#### Performance
```
=== Evaluating at cutoff 0.55 ===
points     14590
positives  1103
accuracy   0.9419
precision  0.7025
recall     0.4025
f1         0.5118
auc        0.9251
```

### Calibration (temperature, then Platt)
Both rescale the output after training without touching the weights: take the logit, compute `a * logit + b`, re-apply the sigmoid. Both are fitted on dev by minimizing NLL.
- temperature scaling: b = 0, a = 1/T (T < 1 sharpens, T > 1 softens)
- Platt scaling: fits a and b, so it can also shift every score up or down

With a > 0 both are monotonic, so the ranking (and AUC) doesn't change, only what a score *means*.

`CALIBRATION` picks the method (`"platt"`). The fit is saved as `calibration.json` next to the weights, and `predict`/`inference` apply it (so `Program.cs` gets calibrated scores too). Training fits it at the end. For an existing model: `python machine_learning.py calibrate <weights> <token_map> [platt|temperature]`. If there's no `calibration.json`, scores are left raw.

For `turns` on dev (base rate 0.076, median raw score 0.009):
```
method       a       b       NLL     ECE
raw          1       0       0.1510  0.0099
temperature  0.924   0       0.1503  0.0063   (T = 1.082)
platt        0.979   0.206   0.1498  0.0036
```
Temperature came out *above* 1, which softens. The huge 0.0-0.1 bin, where the model is slightly overconfident, sets T, while the model is *under*confident from 0.2 to 0.8. Platt's b shifts everything up instead, which fixes the middle:
```
raw bin   n      raw    temp   platt  actual
0.0-0.1   12647  0.015  0.019  0.019  0.019
0.1-0.2   534    0.143  0.160  0.175  0.169
0.2-0.3   289    0.246  0.262  0.291  0.332
0.3-0.4   204    0.348  0.359  0.399  0.392
0.4-0.5   161    0.450  0.454  0.502  0.522
0.5-0.6   268    0.552  0.548  0.601  0.578
0.6-0.7   294    0.645  0.635  0.688  0.677
0.7-0.8   149    0.753  0.738  0.786  0.826
0.8-0.9   44     0.811  0.794  0.837  0.750
```
The 0.8-0.9 bin is still overconfident, but it's only 44 points. Isotonic regression would be the next step if that matters.

`ROUTING_CUTOFF` 0.55 -> 0.599, the Platt equivalent of raw 0.55, so the routing decisions stay the same:
```
=== Evaluating at cutoff 0.599 ===
points     14590
positives  1103
accuracy   0.9418
precision  0.7003
recall     0.4025
f1         0.5112
auc        0.9248
```
