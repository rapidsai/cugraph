#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION.
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
if __name__ == "__main__":
    import argparse

    from nx_cugraph.scripts import print_table, print_tree

    parser = argparse.ArgumentParser(
        parents=[
            print_table.get_argumentparser(add_help=False),
            print_tree.get_argumentparser(add_help=False),
        ],
        description="Print info about functions implemented by nx-cugraph",
    )
    parser.add_argument("action", choices=["print_table", "print_tree"])
    args = parser.parse_args()
    if args.action == "print_table":
        print_table.main()
    else:
        print_tree.main(
            by=args.by,
            networkx_path=args.networkx_path,
            dispatch_name=args.dispatch_name or args.dispatch_name_always,
            version_added=args.version_added,
            plc=args.plc,
            dispatch_name_if_different=not args.dispatch_name_always,
        )
