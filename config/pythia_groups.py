"""Configuration for Pythia model attention head groups."""

PYTHIA_GROUPS = [
    {
        "name": "Subject Heads",
        "vertices": [[17, 2], [16, 12], [21, 9], [16, 20], [22, 17], [18, 14]],
        "description": "Attend to subject tokens and extract their attributes. May activate even when irrelevant to the query."
    },
    {
        "name": "Relation Heads",
        "vertices": [[13, 31], [18, 20], [14, 24], [21, 18]],
        "description": "Focus on relation tokens and boost possible answers for that relation type. Operate independently of subjects."
    },
    {
        "name": "Mixed Heads",
        "vertices": [[17, 17], [21, 23], [23, 22], [26, 8], [22, 15], [17, 30], [18, 25]],
        "description": "Attend to both subject and relation tokens. Extract correct attributes more effectively through \"subject to relation propagation.\""
    }
] 