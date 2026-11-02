# Error codes

Format: `[LV-E101] message. Hint: ...`

| Code | Class | Situation |
|---|---|---|
| LV-E101 | LvDataError | column not found |
| LV-E102 | LvDataError | wrong dtype |
| LV-E103 | LvDataError | empty after NaN drop |
| LV-E104 | LvDataError | duplicate columns |
| LV-E106 | LvDataError | y_true not binary |
| LV-E107 | LvDataError | upset sets 2-20 |
| LV-E108 | LvDataError | file not found |
| LV-E109 | LvDataError | unsupported format |
| LV-E110 | LvDataError | URL unreachable |
| LV-E201 | LvArgumentError | out of range |
| LV-E202 | LvArgumentError | wrong type |
| LV-E203 | LvArgumentError | unknown theme |
| LV-E204 | LvArgumentError | incompatible args |
| LV-E205 | LvDataError | >30 groups |
| LV-E301 | LvExportError | unsupported export |
| LV-E302 | LvExportError | no write permission |
| LV-E310 | LvArgumentError | backend unavailable |
| LV-E401 | LvDependencyError | missing optional dep |
| LV-W501 | LvDataWarning | NaN rows dropped |
| LV-W502 | LvDataWarning | subsampled points |
| LV-W503 | LvPerformanceWarning | large data |
| LV-W504 | LvDataWarning | stats skipped |
