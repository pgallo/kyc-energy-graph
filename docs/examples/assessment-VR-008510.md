# Counterparty assessment: VR-008510

**Fjelarovrik Engineering & Construction PLC** (synthetic data, seed 7, as of 2026-06-30)

| risk | band | confidence | review |
|---|---|---|---|
| 0.986 | high | 0.950 | risk 0.99 >= 0.3 |

## Entity

Resolved from 3 record(s) in sources A, B, REG: `REG:VR-008510`, `A:A00171`, `B:B00157`

| attribute | value |
|---|---|
| jurisdiction | VA |
| address | 17 Lipova Street, Dornhavn |
| incorporated | 2001-01-13 |
| sector | EN42.22 |
| employees | 298 |
| reg_no | VR-008510 |

## Ownership tree

Shares are fused across sources; `eff` is effective (integrated) ownership in this company.

```
VR-008510  Fjelarovrik Engineering & Construction PLC
├── 55.0% VR-008473  Fjelareth Capital Ltd [VA]  eff 55.0%  src A,REG
│   ├── 59.9% X-A-A00169  Fjelaraltor Investments Inc [MR]  eff 32.9%  src A,B
│   │   ├── 69.5% X-B-B00400  Zoran Kostenenko [KS]  eff 22.9%  src B
│   │   └── 30.0% X-A-A00463  Timur Poloich [MR]  eff 9.9%  src A
│   └── 40.0% NID-2728987  Radan Ostenchuk [VA]  eff 22.0%  src A,REG
└── 45.0% NID-1577920  Selma Zhelovets [VA]  eff 45.0%  src REG
```

## Key persons

| person | name | roles | effective | influence | betweenness | bridge | nominee pattern | screening |
|---|---|---|---|---|---|---|---|---|
| NID-1577920 | Selma Zhelovets | ubo | 45.0% | 1.000 | 0.0059 | False | False | none |
| X-B-B00400 | Zoran Kostenenko | controller | 22.9% | 1.000 | 0.000 | False | False | SL-0001 hit (0.99) |
| NID-3887761 | Petra Velanski | director | 0.0% | 0.600 | 0.000 | False | False | none |
| NID-6030126 | Irena Tomov | director | 0.0% | 0.600 | 0.012 | False | False | none |
| NID-8663575 | Lumira Shevarski | parent_director | 0.0% | 0.500 | 0.0020 | False | False | none |
| NID-1845231 | Marek Ostyev | signatory | 0.0% | 0.400 | 0.000 | False | False | none |

## Evidence and fusion

Fusion in log-odds from prior 3% (logit -3.48). Each item contributes confidence x source weight x ln(LR); only the strongest item per kind is counted.

| kind | LR | confidence | source weight | contribution | counted | summary |
|---|---|---|---|---|---|---|
| key_person_screening_sanctions | 40.000 | 0.991 | 0.950 | 3.472 | yes | via X-B-B00400 (controller): Matches SL-0001 'ZORAN KOSTENENKO' (hit, name 1.00, dob +1.0) |
| sanctioned_exposure | 20.000 | 0.760 | 1.000 | 2.276 | yes | X-B-B00400 (Zoran Kostenenko) matches SL-0001 (conf 0.99); 3 hop(s), effective 23%, controlling |
| ml_counterparty_model | 20.000 | 0.800 | 0.500 | 1.198 | yes | Gradient-boosting score 0.91; top drivers: lr_sanctioned_exposure=2.28 (+0.78), kp_sanctions_conf=0.99 (+0.52), tender_awards=1.00 (+0.01), lr_hidden_ubo=0.00 (+0.00) |
| key_person_adverse_media | 5.000 | 0.800 | 0.600 | 0.773 | yes | via X-B-B00400 (controller): M004 (2026-01-15): "Investigators allege Zoran Kostenenko took part in a sanctions evasion scheme" |

Posterior logit +4.24 gives risk 0.986.

### Evidence chains

1. **sanctioned_exposure** (sources: A, B, REG, lists)
   - edge: `X-B-B00400 -[70% via B]-> X-A-A00169`
   - edge: `X-A-A00169 -[60% via A,B]-> VR-008473`
   - edge: `VR-008473 -[55% via A,REG]-> VR-008510`
   - records: `B:B00400`, `C:C00391`, `A:A00169`, `B:B00155`, `REG:VR-008473`, `A:A00170`, `B:B00156`, `REG:VR-008510`, `A:A00171`, `B:B00157`
2. **key_person_screening_sanctions** (sources: lists)
   - edge: `X-B-B00400 -[controller]-> VR-008510`
   - records: `SL-0001`
3. **key_person_adverse_media** (sources: media)
   - edge: `X-B-B00400 -[controller]-> VR-008510`
   - records: `M004`

## Learned counterparty model

Gradient boosting (calibrated) probability 0.910. Largest drivers by occlusion (probability change when the feature is reset to the training median):

| feature | value | effect |
|---|---|---|
| lr_sanctioned_exposure | 2.276 | 0.783 |
| kp_sanctions_conf | 0.991 | 0.520 |
| tender_awards | 1.000 | 0.0085 |
| lr_hidden_ubo | 0.000 | 0.000 |

## Confidence

Confidence is separate from risk: it measures how far the assessment can be trusted.

| component | value |
|---|---|
| data_quality | 1.000 |
| provider_agreement | 1.000 |
| resolution | 0.933 |
| evidence_quality | 0.838 |
| overall | 0.950 |
