# Feature Labels Report

**Total unique features**: 440
**Labeled**: 435 (98.9%)
**Source**: mwhanna/qwen3-4b-transcoders dashboard data (top logits + activation examples)

## Sign-Flipped Features

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L27:F139980 | 'visor', 'igaret', '/fw', ' вещ', ' Explorer' | 0.0015 | 5 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L18:F148614 | ' Ug', 'iques', ' Eag', '…⏎⏎⏎⏎', 'éré' | 0.0153 | 5 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L29:F6866 | '焯', 'anness', '-court', '酩', ' Fathers' | 0.0056 | 5 | analytical, cognitive_reframe, completion, roleplay |
| L13:F65439 | 'ackson', 'ilon', ' Orlando', " '-')", "='-" | 0.0001 | 5 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L27:F6642 | '眨眼', '簇', 'vere', 'sembly', ' \u200b\u200b' | 0.0330 | 4 | analytical, cognitive_reframe, completion, roleplay |
| L6:F30108 | ' resides', '基调', 'ichern', '企图', ' Enterprises' | 0.0041 | 4 | analytical, completion, roleplay |
| L27:F27345 | 'Prompt', ' prompts', 'prompt', ' Prompt', ' pr... | 0.0035 | 4 | analytical, cognitive_reframe, completion |
| L12:F160449 | 'cales', '_emb', 'GEST', '敞', ' //</' | 0.0001 | 4 | analytical, cognitive_reframe, completion, fiction |
| L4:F61314 | 'いま', ' match', 'arning', ' matches', ' implic' | 0.0001 | 3 | analytical, cognitive_reframe, roleplay |
| L12:F157768 | 'strup', 'inke', '叟', 'yne', ' manipulate' | 0.0017 | 3 | cognitive_reframe, completion, roleplay |
| L25:F23176 | '赤', 'able', '饮水', 'ahn', 'azzi' | 0.0004 | 3 | completion, fiction, roleplay |
| L19:F6087 | ' nowrap', ' (*((', 'itol', 'بس', '辫' | 0.0004 | 3 | completion, roleplay |
| L31:F86700 |  | N/A | 3 | analytical, completion, roleplay |
| L13:F110090 | '服务质量', ' olmasını', 'isor', '惭', 'llib' | 0.0055 | 3 | analytical, fiction, roleplay |
| L18:F51630 | '一体化', '经济效益', 'Simon', '最长', ' Kahn' | 0.0016 | 3 | cognitive_reframe, completion, roleplay |

## Dampened Features (pro-refusal weakened by JB)

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L32:F132684 | ' Sol', ' Making', 'Navig', ' Expl', ' Tre' | 0.0078 | 16 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L29:F18221 | 'Unfortunately', '遗憾', ' Unfortunately', '不幸', ... | 0.0023 | 14 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L31:F109901 | ' engages', ' engaging', ' engagement', ' engag... | 0.0060 | 14 | analytical, cognitive_reframe, completion, fiction |
| L31:F49541 | '我会', '我也', 'I', '我', ' I' | 0.0049 | 11 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L31:F116260 | ' spam', '麵', '垃圾', '翻', '盗' | 0.0011 | 9 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L33:F67079 | '尊重', 'res', ' Res', ' respect', ' res' | 0.0005 | 9 | analytical, cognitive_reframe, completion, roleplay |
| L28:F160311 | '違', 'illegal', '违法', '非法', ' illegal' | 0.0035 | 9 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L28:F87125 | ' providing', '提供', '出示', '展示', ' offering' | 0.0015 | 9 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L29:F64904 | '霸', ' bully', ' Bul', ' bull', 'bul' | 0.0049 | 8 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L33:F78685 | ' numbering', ' recogn', '开花', '-leaning', '→lo... | 0.0006 | 8 | analytical, completion, fiction, roleplay |
| L32:F71061 | ' supporting', ' enabling', ' helping', ' facil... | 0.0085 | 8 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L32:F158432 | '创造', 'create', '创', '创建', ' create' | 0.0011 | 7 | analytical, completion, fiction, roleplay |
| L26:F82435 | ' hatred', '解决', '霸王', ' cheat', ' greed' | 0.0028 | 6 | analytical, cognitive_reframe, completion, fiction |
| L29:F152160 | ' eth', ' ethics', '道德', ' Eth', ' ethical' | 0.0068 | 6 | analytical, cognitive_reframe, fiction |
| L28:F101616 | ' inserting', ' putting', ' applying', ' removi... | 0.0052 | 6 | analytical, cognitive_reframe, completion, fiction, roleplay |

## Amplified Anti-Refusal Features

| Feature | Top Logits | Freq | Count | Classes |
|---------|-----------|------|-------|---------|
| L32:F36682 | 'onte', 'ü', 'indrome', '↵⏎↵⏎', 'omial' | 0.0005 | 12 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L17:F82119 | '生活环境', 'resses', '�认', 'ья', 'ogi' | 0.0019 | 11 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L24:F81123 | '当然', 'Sure', 'Certainly', 'sure', ' Certainly' | 0.0006 | 9 | analytical, completion, fiction, roleplay |
| L16:F35905 | 'rates', ' Attention', 'uil', '穷', '个项目' | 0.0005 | 9 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L18:F161543 | '入', '悬念', '出路', '咬', '游' | 0.0018 | 7 | analytical, cognitive_reframe, completion, fiction, roleplay |
| L32:F79442 |  | N/A | 6 | analytical, fiction |
| L31:F85768 | ' quando', ' When', '→when', 'when', ' when' | 0.0028 | 5 | analytical, cognitive_reframe, roleplay |
| L16:F132756 | 'ibu', ' następn', 'riet', 'rido', '每一天' | 0.0006 | 5 | analytical, cognitive_reframe, completion |
| L33:F97879 | 'validation', 'ccb', ' Converted', '[opt', ' ph... | 0.0098 | 5 | analytical, completion, roleplay |
| L27:F84764 | 'Here', 'here', ' Here', '这里', ' here' | 0.0067 | 4 | analytical, fiction |
| L22:F153605 | '的动力', ' Fritz', '成就', ' STAR', '青山' | 0.0009 | 4 | cognitive_reframe, completion, fiction |
| L19:F88090 | ' Oasis', '薪', 'umber', '确定', '请点击' | 0.0007 | 3 | completion, fiction, roleplay |
| L29:F114615 | '肯定', ' absolutely', ' Definitely', ' certainly... | 0.0023 | 3 | analytical, fiction, roleplay |
| L27:F857 | 'iku', ' pos', 'oux', ' przec', 'ucid' | 0.0003 | 2 | cognitive_reframe |
| L31:F65687 | ' turb', ' Owen', '离', '螺', ' centrif' | 0.0009 | 2 | completion, roleplay |

