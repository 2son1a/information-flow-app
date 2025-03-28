"""Configuration for GPT-2 model attention head groups."""

GPT2_GROUPS = [
    {
        "name": "Name Mover",
        "vertices": [[9, 9], [10, 0], [9, 6]],
        "description": "Attend to names and copy them to output. Active at END token position."
    },
    {
        "name": "Negative",
        "vertices": [[10, 7], [11, 10]],
        "description": "Write in opposite direction of Name Movers, decreasing prediction confidence."
    },
    {
        "name": "S Inhibition",
        "vertices": [[8, 10], [7, 9], [8, 6], [7, 3]],
        "description": "Reduce Name Mover Heads' attention to subject tokens. Attend to S2 and modify query patterns."
    },
    {
        "name": "Induction",
        "vertices": [[5, 5], [5, 9], [6, 9], [5, 8]],
        "description": "Recognize [A][B]...[A] patterns to detect duplicated tokens via different mechanism."
    },
    {
        "name": "Duplicate Token",
        "vertices": [[0, 1], [0, 10], [3, 0]],
        "description": "Identify repeated tokens. Active at S2, attend to S1, signal token duplication."
    },
    {
        "name": "Previous Token",
        "vertices": [[4, 11], [2, 2]],
        "description": "Copy subject information to the token after S1. Support Induction Heads."
    },
    {
        "name": "Backup Name Mover",
        "vertices": [[11, 2], [10, 6], [10, 10], [10, 2], [9, 7], [10, 1], [11, 9], [9, 0]],
        "description": "Normally inactive but replace Name Movers if they're disabled. Show circuit redundancy."
    }
] 