#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
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

import paddle.fluid.incubate.data_generator as dg

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class DacDataset(dg.MultiSlotDataGenerator):
    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            features = line.rstrip('\n').split('\t')
            dense_feature = []
            sparse_feature = []
            for idx in continuous_range_:
                if features[idx] == "":
                    dense_feature.append(0.0)
                else:
                    dense_feature.append(
                        (float(features[idx]) - cont_min_[idx - 1]) /
                        cont_diff_[idx - 1])
            for idx in categorical_range_:
                sparse_feature.append(
                    [hash(str(idx) + features[idx]) % hash_dim_])
            label = [int(features[0])]
            process_line = dense_feature, sparse_feature, label
            feature_name = ["dense_feature"]
            for idx in categorical_range_:
                feature_name.append("C" + str(idx - 13))
            feature_name.append("label")

            yield zip(feature_name, [dense_feature] + sparse_feature + [label])

        return reader


d = DacDataset()
d.run_from_stdin()