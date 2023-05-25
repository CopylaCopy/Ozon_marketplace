import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict
from tqdm import tqdm


class Splitter:
    def __init__(self, pairs: pd.DataFrame):
        self.component_to_id = defaultdict(lambda: set())
        self.id_to_component = {}

        curr_component = 1

        for i, row in enumerate(tqdm(pairs.iterrows(), desc="Calculating components")):
            id_1 = int(row[1].variantid1)
            id_2 = int(row[1].variantid2)

            id_1_component = self.id_to_component.get(id_1)
            id_2_component = self.id_to_component.get(id_2)

            if (id_1_component and id_2_component) and (
                id_1_component != id_2_component
            ):
                # print(f"Merging {id_1_component} and {id_2_component}")
                self.component_to_id[id_1_component].update(
                    self.component_to_id[id_2_component]
                )

                for ids in self.component_to_id[id_2_component]:
                    self.id_to_component[ids] = id_1_component

                self.component_to_id.pop(id_2_component)

            elif id_1_component or id_2_component:
                component = id_1_component or id_2_component
                self.component_to_id[component].update({id_1, id_2})

                self.id_to_component[id_1] = component
                self.id_to_component[id_2] = component

            else:
                assert self.component_to_id.get(curr_component) is None, "SHIT"
                self.component_to_id[curr_component] = {id_1, id_2}
                self.id_to_component[id_1] = curr_component
                self.id_to_component[id_2] = curr_component

                curr_component += 1

    def split(self, data, test_size, random_state=None):
        groups = []

        for row in tqdm(data.iterrows(), desc="Adding component numbers"):
            id_1 = int(row[1].variantid1)
            id_2 = int(row[1].variantid2)

            assert self.id_to_component[id_1] == self.id_to_component[id_2]

            groups.append(self.id_to_component[id_1])

        splitter = GroupShuffleSplit(
            test_size=test_size, n_splits=1, random_state=random_state
        )
        split = splitter.split(groups, groups=groups)
        train_inds, test_inds = next(split)

        train = data.iloc[train_inds]
        test = data.iloc[test_inds]

        return train, test


def stratified_group_split(pairs, test_size, random_state=None):
    splitter = Splitter(pairs)
    return splitter.split(pairs, test_size=test_size, random_state=random_state)
