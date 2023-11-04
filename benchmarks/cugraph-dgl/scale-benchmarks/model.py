# Copyright (c) 2018-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import time


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        model_backend="dgl",
    ):
        if model_backend == "dgl":
            from dgl.nn import SAGEConv
        else:
            from cugraph_dgl.nn import SAGEConv

        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SAGEConv(in_channels, hidden_channels, aggregator_type="mean")
            )
            in_channels = hidden_channels
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, aggregator_type="mean")
        )

    def forward(self, blocks, x):
        for i, conv in enumerate(self.convs):
            x = conv(blocks[i], x)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x


def create_model(feat_size, num_classes, num_layers, model_backend="dgl"):
    model = GNN(feat_size, 64, num_classes, num_layers, model_backend=model_backend)
    model = model.to("cuda")
    model.train()
    return model


def train_model(model, dataloader, opt, feat, y):
    times = {key: 0 for key in ["mfg_creation", "feature", "m_fwd", "m_bkwd"]}
    epoch_st = time.time()
    mfg_st = time.time()
    for input_nodes, output_nodes, blocks in dataloader:
        times["mfg_creation"] += time.time() - mfg_st
        if feat is not None:
            fst = time.time()
            input_nodes = input_nodes.to("cpu")
            input_feat = feat[input_nodes]
            input_feat = input_feat.to("cuda")
            if isinstance(output_nodes, dict):
                output_nodes = output_nodes["paper"]
            output_nodes = output_nodes.to(y.device)
            y_batch = y[output_nodes].to("cuda")
            times["feature"] += time.time() - fst

            m_fwd_st = time.time()
            y_hat = model(blocks, input_feat)
            times["m_fwd"] += time.time() - m_fwd_st

            m_bkwd_st = time.time()
            loss = F.cross_entropy(y_hat, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            times["m_bkwd"] += time.time() - m_bkwd_st
        mfg_st = time.time()

    print(f"Epoch time = {time.time() - epoch_st:.2f} seconds")

    return times


def analyze_time(dataloader, times, epoch_time, fanout, batch_size):
    num_batches = len(dataloader)
    time_d = {
        "fanout": fanout,
        "batch_size": batch_size,
        "epoch_time": epoch_time,
        "epoch_time_per_batch": epoch_time / num_batches,
        "num_batches": num_batches,
    }
    for key, value in times.items():
        time_d[f"{key}_time_per_epoch"] = value
        time_d[f"{key}_time_per_batch"] = value / num_batches

    print(f"Time analysis for fanout = {fanout}, batch_size = {batch_size}")
    for k in time_d.keys():
        if "time_per_epoch" in str(k):
            print(f"{k} = {time_d[k]:.2f} seconds")
    return time_d


def run_1_epoch(dataloader, feat, y, fanout, batch_size, model_backend):
    if feat is not None:
        model = create_model(
            feat.shape[1], 172, len(fanout), model_backend=model_backend
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
    else:
        model = None
        opt = None

    # Warmup RUN
    times = train_model(model, dataloader, opt, feat, y)

    epoch_st = time.time()
    times = train_model(model, dataloader, opt, feat, y)
    epoch_time = time.time() - epoch_st
    time_d = analyze_time(dataloader, times, epoch_time, fanout, batch_size)
    return time_d
