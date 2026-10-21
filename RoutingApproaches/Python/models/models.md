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
