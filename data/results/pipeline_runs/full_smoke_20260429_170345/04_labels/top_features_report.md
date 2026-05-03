# Feature Labels Report

**Total unique features**: 379
**Labeled**: 379 (100.0%)
**Source**: mwhanna/gemma-scope-2-4b-it dashboard data (top logits + activation examples)

## Sign-Flipped Features

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L13:F970 | 'mr', ' ένα', 'Mr', 'rule', 'sq' | 0.0061 | 17 | ctrl_fiction, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F2106 | ' atm', 'veen', ' मुझे', ' saya', ' please' | 0.0091 | 13 | ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion |
| L14:F178 | ' fruitless', ' hojas', 'astotal', ' snacks', '... | 0.0581 | 11 | ctrl_analytical, ctrl_completion, jb_analytical, jb_cognitive_reframe |
| L14:F64 | 'unsafe', ' опас', ' unsafe', '安全', 'dangerous' | 0.0313 | 11 | jb_analytical, jb_cognitive_reframe, jb_completion |
| L12:F491 | ' Notas', 'Dados', ' Habits', ' Органи', 'Carac... | 0.0750 | 10 | ctrl_analytical, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_roleplay |
| L14:F984 | ' Editar', 'Ნ', ' sekal', ' Dateien', ' authored' | 0.0024 | 10 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_cognitive_reframe |
| L2:F542 | '특별시', ' بأن', '그리고', 'wheeled', '눔' | 0.0186 | 10 | jb_analytical, jb_cognitive_reframe, jb_fiction |
| L12:F8023 | 'checkmark', ' needn', ' homogeneity', '先の', 'မူ' | 0.0052 | 9 | ctrl_fiction, ctrl_roleplay, jb_fiction, jb_roleplay |
| L0:F3997 | '𝘳', '𝘭', 'ির', '𝖒', '𝘮' | 0.0007 | 8 | ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L12:F39 | 'setzungen', ' dihasil', ' evolución', ' evoluç... | 0.0352 | 8 | ctrl_analytical, ctrl_cognitive_reframe, jb_completion, jb_fiction |
| L11:F507 | ' or', '/', 'หรือ', ' and', ' или' | 0.0982 | 8 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_roleplay, jb_completion, jb_fiction, jb_roleplay |
| L10:F439 | ' calcular', 'Argb', 'Calculate', 'Genetic', ' ... | 0.0063 | 8 | jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction |
| L8:F199 | ' করির', 'မြင်', 'စျေး', ' wiel', '䡃' | 0.0667 | 7 | jb_completion, jb_fiction |
| L9:F1060 | 'isfile', 'scapes', 'আমর', ' ибо', ' Fordham' | 0.0041 | 7 | jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction |
| L9:F9095 | 'ach', 'inspiration', 'peak', 'ys', 'Compression' | 0.0007 | 7 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, jb_completion |

## Dampened Features (pro-refusal weakened by JB)

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L10:F234 | ' möglichst', ' absorbance', ' voltages', ' rep... | 0.0227 | 21 | ctrl_cognitive_reframe, ctrl_fiction, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L2:F343 | 'া', 'ldquo', '样的', 'ográfico', 'WeakTable' | 0.0036 | 20 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L9:F11573 | ' fondamentali', ' fondamentale', ' fondament',... | 0.0016 | 20 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L7:F361 | 'opia', ' sleep', ' libs', '?”', '?’' | 0.0240 | 17 | ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction |
| L9:F1080 | ' Ipsum', ' descrizione', ' femenina', ' inform... | 0.0080 | 13 | ctrl_cognitive_reframe, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_fiction |
| L10:F439 | ' calcular', 'Argb', 'Calculate', 'Genetic', ' ... | 0.0063 | 13 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L15:F383 | ' várias', 'Ан', 'Nope', ' verschillende', '੍ਹ' | 0.0068 | 12 | jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction |
| L3:F4750 | 'er', 'ური', 'ीन', 'ার', 'ुरे' | 0.0003 | 12 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, jb_analytical, jb_cognitive_reframe, jb_completion |
| L11:F315 | '3', 'About', "'", '_', ' Re' | 0.0047 | 11 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L3:F29 | 'াধিকার', 'ripcion', ' awfully', ' lecithin', '... | 0.0064 | 11 | ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_cognitive_reframe, jb_fiction, jb_roleplay |
| L13:F427 | ' amic', ' Descent', ' Company', ' Preface', ' ... | 0.0001 | 11 | jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F178 | ' fruitless', ' hojas', 'astotal', ' snacks', '... | 0.0581 | 10 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_roleplay, jb_completion, jb_roleplay |
| L14:F566 | ' gimm', ' अनावश्यक', 'AppendLine', ' mindless'... | 0.0375 | 10 | jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L0:F6994 | 'ट्स', ' incom', 'Ч', 'Veronica', 'ច' | 0.0087 | 9 | jb_analytical, jb_completion, jb_fiction |
| L11:F9217 | ' chronological', ' thirty', ' Thirty', ' vase'... | 0.0013 | 9 | ctrl_analytical, ctrl_cognitive_reframe, jb_analytical, jb_cognitive_reframe, jb_completion, jb_roleplay |

## Amplified Anti-Refusal Features

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L13:F51 | ' vicenda', ' señala', 'elesaian', ' və', ' veí... | 0.0194 | 28 | ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F187 | ' seemingly', ' loosened', ',”', ' minimizes', ... | 0.0358 | 28 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F140 | 'Honestly', 'Handsome', 'Sorry', 'Incorrect', '... | 0.0063 | 20 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L13:F471 | ' Nobel', 'Let', ' hadn', '辖', ' Let' | 0.0028 | 20 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion |
| L14:F541 | ' shun', ' skyscrapers', ' dick', ' laurel', ' ... | 0.0203 | 20 | ctrl_completion, ctrl_fiction, jb_analytical, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L15:F1341 | ' mathematician', ' Donna', ' Preparing', ' MFA... | 0.0264 | 19 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_roleplay |
| L0:F460 | 'quired', 'CH', 'wg', ' ventricle', 'े' | 0.0050 | 17 | ctrl_fiction, ctrl_roleplay, jb_completion, jb_fiction, jb_roleplay |
| L14:F426 | ' huống', '!.', ' family', '顔', '的新' | 0.0035 | 16 | ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_fiction, jb_roleplay |
| L14:F42 | '琊', ' Adolph', ' establecido', ' Dora', ' dora... | 0.0075 | 13 | jb_analytical, jb_cognitive_reframe, jb_fiction, jb_roleplay |
| L15:F476 | 'ំហ', ' pride', ' tolerancia', 'opathy', ' गिफ्ट' | 0.0204 | 13 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_completion |
| L10:F7901 | '.', 'ol', 'ar', ' to', 'me' | 0.0029 | 12 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_completion, jb_roleplay |
| L0:F369 | 'ために', 'т', 'es', 'ための', '𝑡' | 0.0056 | 11 | ctrl_analytical, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_cognitive_reframe, jb_completion, jb_roleplay |
| L14:F480 | ' محدود', ' overhauled', ' cromosoma', 'ฉ', 'เว... | 0.0024 | 11 | ctrl_cognitive_reframe, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_cognitive_reframe, jb_completion, jb_roleplay |
| L15:F24 | ' Cũng', ' Nói', ' ଚ', 'ogène', ' Alors' | 0.0275 | 10 | ctrl_analytical, ctrl_cognitive_reframe, ctrl_completion, ctrl_fiction, ctrl_roleplay, jb_cognitive_reframe, jb_completion, jb_fiction, jb_roleplay |
| L14:F3327 | 'bl', 'tri', 'loop', 'my', ' bl' | 0.0109 | 10 | ctrl_cognitive_reframe, ctrl_fiction, ctrl_roleplay, jb_analytical, jb_fiction |

