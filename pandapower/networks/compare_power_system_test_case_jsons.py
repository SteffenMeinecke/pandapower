import os
import pandapower as pp
from pandapower.networks.create_power_system_test_case_jsons import test_case_table_to_create
import pathlib
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
home = str(pathlib.Path.home())


def compare_json_folders(folder1, folder2):
    """This function allows to compare the results of pandapower networks of different version
    stored as json files in two folders. In folder1, for example, the newly converted jsons files
    exist while in folder2 a copy of the json files of the last relase were stored.
    """
    folder1 = defaulting_folder_integers(folder1)
    folder2 = defaulting_folder_integers(folder2)

    if not os.path.isdir(folder1):
        raise FileNotFoundError(f"This path does not exist as directory: {folder1}")
    if not os.path.isdir(folder2):
        raise FileNotFoundError(f"This path does not exist as directory: {folder2}")

    test_case_df, _ = test_case_table_to_create()
    net_names = list(test_case_df.name)
    net_names = ["case33bw", "case300", "case6470rte", "case6495rte", "case6515rte"] # TODO for testing

    eq = list()
    not_eq = list()
    load_err = list()

    for net_name in net_names:
        try:
            net1 = pp.from_json(os.path.join(folder1, f"{net_name}.json"))
            net2 = pp.from_json(os.path.join(folder2, f"{net_name}.json"))
        except:
            load_err.append(net_name)
        else:
            if 1:
                et_cts1 = pp.count_elements(net1)
                et_cts2 = pp.count_elements(net2)
                for et in et_cts1.index.union(et_cts2.index):
                    cols_to_ignore = ["name"]
                    if "et" == "load":
                        cols_to_ignore.append("type")
                    df_eq = pp.dataframes_equal(net1[et][net1[et].columns.difference({"name"})],
                                                net2[et][net1[et].columns.difference({"name"})])
                    print(f"{et}: {df_eq}")
            is_eq = pp.nets_equal(net1, net2, check_only_results=True)
            if is_eq:
                eq.append(net_name)
            else:
                not_eq.append(net_name)

    if len(load_err):
        logger.error(f"These networks could not be loaded from json files:\n{load_err}")
    if len(not_eq):
        logger.error(f"The comparison reveald result differences for these networks:\n{not_eq}")

    return eq, not_eq, load_err


def defaulting_folder_integers(folder):
    return folder if isinstance(folder, str) else \
        os.path.join(home, "Desktop", f"power_system_test_case_jsons_{folder}")


if __name__ == "__main__":
    compare_json_folders(1, os.path.join(home, "desktop", "develop"))
