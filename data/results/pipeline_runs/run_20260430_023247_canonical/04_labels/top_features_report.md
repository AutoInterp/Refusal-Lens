# Feature Labels Report

**Total unique features**: 1149
**Labeled**: 1149 (100.0%)
**Source**: mwhanna/gemma-scope-2-4b-it dashboard data (top logits + activation examples)

## Sign-Flipped Features

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L13:F970 | 'mr', ' ένα', 'Mr', 'rule', 'sq' | 0.0061 | 184 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F1779 | ' El', ' prime', 'ებმა', ' பின்', ' el' | 0.0205 | 178 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F2106 | ' atm', 'veen', ' मुझे', ' saya', ' please' | 0.0091 | 175 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L11:F507 | ' or', '/', 'หรือ', ' and', ' или' | 0.0982 | 171 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F327 | 'DebuggerNonUser', ' devoured', ' Symfony', ' t... | 0.0301 | 128 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F178 | ' fruitless', ' hojas', 'astotal', ' snacks', '... | 0.0581 | 106 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_roleplay |
| L13:F1163 | 'rater', 'ශ්', 'Carpenter', 'Romantic', 'RMSE' | 0.0045 | 104 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L12:F491 | ' Notas', 'Dados', ' Habits', ' Органи', 'Carac... | 0.0750 | 103 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F984 | ' Editar', 'Ნ', ' sekal', ' Dateien', ' authored' | 0.0024 | 103 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion |
| L13:F6636 | ' an', ' gering', ' prescribed', ' 제한', ' higher' | 0.0005 | 101 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F1001 | 'encia', 'asku', ' ﬁ', 'enciais', '𝟖' | 0.0238 | 95 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_roleplay |
| L9:F1060 | 'isfile', 'scapes', 'আমর', ' ибо', ' Fordham' | 0.0041 | 89 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F497 | ' Quadrupèdes', ' უფრო', ' Safer', 'ಸ್ಪ', 'SAFER' | 0.0748 | 85 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L12:F8023 | 'checkmark', ' needn', ' homogeneity', '先の', 'မူ' | 0.0052 | 84 | ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L2:F542 | '특별시', ' بأن', '그리고', 'wheeled', '눔' | 0.0186 | 82 | ctrl_completion, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |

## Dampened Features (pro-refusal weakened by JB)

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L7:F361 | 'opia', ' sleep', ' libs', '?”', '?’' | 0.0240 | 322 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F178 | ' fruitless', ' hojas', 'astotal', ' snacks', '... | 0.0581 | 290 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L10:F234 | ' möglichst', ' absorbance', ' voltages', ' rep... | 0.0227 | 277 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L11:F315 | '3', 'About', "'", '_', ' Re' | 0.0047 | 231 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L2:F343 | 'া', 'ldquo', '样的', 'ográfico', 'WeakTable' | 0.0036 | 221 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L9:F1080 | ' Ipsum', ' descrizione', ' femenina', ' inform... | 0.0080 | 219 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction |
| L10:F439 | ' calcular', 'Argb', 'Calculate', 'Genetic', ' ... | 0.0063 | 192 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L15:F383 | ' várias', 'Ан', 'Nope', ' verschillende', '੍ਹ' | 0.0068 | 190 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L9:F11573 | ' fondamentali', ' fondamentale', ' fondament',... | 0.0016 | 181 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F427 | ' amic', ' Descent', ' Company', ' Preface', ' ... | 0.0001 | 165 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F566 | ' gimm', ' अनावश्यक', 'AppendLine', ' mindless'... | 0.0375 | 160 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L0:F6994 | 'ट्स', ' incom', 'Ч', 'Veronica', 'ច' | 0.0087 | 140 | jb_analytical, jb_completion, jb_fiction |
| L10:F259 | ' syrups', ' necessitating', 'ગાહી', ' sultry',... | 0.0162 | 139 | ctrl_analytical, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L3:F196 | 'ஜ்மஹால்', ' diphtheria', '𝙩', ' enda', 'łem' | 0.0056 | 122 | ctrl_analytical, ctrl_completion, jb_analytical, jb_completion, jb_roleplay |
| L3:F4750 | 'er', 'ური', 'ीन', 'ার', 'ुरे' | 0.0003 | 120 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, jb_analytical, jb_cognitive_reframe, jb_completion |

## Amplified Anti-Refusal Features

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L14:F187 | ' seemingly', ' loosened', ',”', ' minimizes', ... | 0.0358 | 405 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F140 | 'Honestly', 'Handsome', 'Sorry', 'Incorrect', '... | 0.0063 | 321 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F471 | ' Nobel', 'Let', ' hadn', '辖', ' Let' | 0.0028 | 291 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_roleplay |
| L14:F426 | ' huống', '!.', ' family', '顔', '的新' | 0.0035 | 291 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_fiction, jb_roleplay |
| L15:F1341 | ' mathematician', ' Donna', ' Preparing', ' MFA... | 0.0264 | 267 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_roleplay |
| L13:F51 | ' vicenda', ' señala', 'elesaian', ' və', ' veí... | 0.0194 | 248 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L0:F460 | 'quired', 'CH', 'wg', ' ventricle', 'े' | 0.0050 | 214 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F3327 | 'bl', 'tri', 'loop', 'my', ' bl' | 0.0109 | 211 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F42 | '琊', ' Adolph', ' establecido', ' Dora', ' dora... | 0.0075 | 189 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L15:F476 | 'ំហ', ' pride', ' tolerancia', 'opathy', ' गिफ्ट' | 0.0204 | 179 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_completion, jb_fiction, jb_roleplay |
| L14:F541 | ' shun', ' skyscrapers', ' dick', ' laurel', ' ... | 0.0203 | 166 | ctrl_completion, ctrl_fiction, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F167 | '้ง', 'DUCTION', 'ware', 'هرات', '่วง' | 0.0175 | 166 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_fiction, jb_roleplay |
| L13:F6712 | ' juiste', ' Most', 'Most', 'الان', ' 위의' | 0.0086 | 153 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F1402 | 'いただきます', ' παρά', ' 상세', 'ndrome', ' れる' | 0.0170 | 150 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F311 | ' gemäß', 'neither', 'без', 'Neither', ' gaji' | 0.0049 | 149 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_roleplay |

