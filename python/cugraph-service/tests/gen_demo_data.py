# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import random

###############################################################################
# vertex CSV
colors = ["red", "white", "blue", "green", "yellow", "orange", "black", "purple"]

with open("vertex_data.csv", "w") as vertex_out:
    print("vertex_id color num_stars", file=vertex_out)

    for i in range(1000):
        print(
            f"{i} {random.choice(colors)} {int(random.random() * 10000)}",
            file=vertex_out,
        )


###############################################################################
# edge CSV
relationship = ["friend", "coworker", "reviewer"]
ids = range(1000)

with open("edge_data.csv", "w") as edge_out:
    print("src dst relationship_type num_interactions", file=edge_out)

    for i in range(10000):
        src = random.choice(ids)
        dst = random.choice(ids)
        while src == dst:
            dst = random.choice(ids)

        print(
            f"{src} {dst} "
            f"{random.choice(relationship)} "
            f"{int((random.random() + 1) * 10)}",
            file=edge_out,
        )
