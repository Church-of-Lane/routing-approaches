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
