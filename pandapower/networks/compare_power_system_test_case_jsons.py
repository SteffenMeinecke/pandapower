import os
import pandapower as pp
from pandapower.networks.create_power_system_test_case_jsons import test_case_table_to_create
import pathlib

home = str(pathlib.Path.home())


def compare_json_folders(folder1, folder2):
    """This function allows to compare the results of pandapower networks of different version
    stored as json files in two folders. In folder1, for example, the newly converted jsons files
    exist while in folder2 a copy of the json files of the last relase were stored.
    """
    folder1 = defaulting_folder_integers(folder1)
    folder2 = defaulting_folder_integers(folder2)

    test_case_df, _ = test_case_table_to_create()
    net_names = list(test_case_df.name)

    eq = list()
    not_eq = list()
    load_err = list()

    for net_name in net_names:
        try:
            net1 = pp.from_json(os.path.join(folder1, f"{net_name}.json"))
            net2 = pp.from_json(os.path.join(folder1, f"{net_name}.json"))
        except:
            load_err.append(net_name)
        else:
            is_eq = pp.nets_equal(net1, net2, check_only_results=True)
            if is_eq:
                eq.append(net_name)
            else:
                not_eq.append(net_name)


def defaulting_folder_integers(folder):
    return folder if isinstance(folder, str) else \
        os.path.join(home, "Desktop", f"power_system_test_case_jsons_{folder}")


if __name__ == "__main__":
    compare_json_folders(1, 2)
