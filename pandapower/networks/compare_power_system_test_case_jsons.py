import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
from pandapower.networks.create_power_system_test_case_jsons import test_case_table_to_create
import scipy
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
home = str(pathlib.Path.home())


def compare_json_folders(folder1, folder2, matpower_excel_result_folder=None, mat_file_folder=None):
    """This function allows to compare the results of pandapower networks of different version
    stored as json files in two folders. In folder1, for example, the newly converted jsons files
    exist while in folder2 a copy of the json files of the last relase were stored.
    If matpower_excel_result_folder is given, voltage results are considered and compared, as well.
    """
    folder1 = defaulting_folder_integers(folder1)
    folder2 = defaulting_folder_integers(folder2)

    if not os.path.isdir(folder1):
        raise FileNotFoundError(f"This path does not exist as directory: {folder1}")
    if not os.path.isdir(folder2):
        raise FileNotFoundError(f"This path does not exist as directory: {folder2}")

    test_case_df, _ = test_case_table_to_create()
    net_names = list(test_case_df.name)
    # net_names = ["case14", "case300", "case6470rte", "case6495rte", "case6515rte"] # for testing

    eq = list()
    not_eq = list()
    load_err = list()


    for net_name in net_names:

        # --- compare pp networks from different folders
        try:
            net1 = pp.from_json(os.path.join(folder1, f"{net_name}.json"))
            net2 = pp.from_json(os.path.join(folder2, f"{net_name}.json"))
        except:
            load_err.append(net_name)
        else:
            if 1:
                et_cts1 = pp.count_elements(net1)
                et_cts2 = pp.count_elements(net2)
                print("Number of elements per type:")
                print(pd.concat([et_cts1, et_cts2], axis=1, keys=["net1", "net2"]))
                for i_et, et in enumerate(et_cts1.index.intersection(et_cts2.index)):
                    if i_et == 0:
                        print("Not matching element tables:")
                    cols = net1[et].columns.intersection(net2[et].columns)
                    cols = cols.difference(["name"])
                    if "et" == "load":
                        cols = cols.difference(["type"])
                    df_eq = pp.dataframes_equal(net1[et][cols], net2[et][cols])
                    print(f"{et}: {df_eq}")

            is_eq = pp.nets_equal(net1, net2, check_only_results=True)

            if not is_eq:
                if not net1.converged:
                    try:
                        pp.runpp(net1, trafo_model="pi")
                    except pp.LoadflowNotConverged:
                        pass
                if not net2.converged:
                    try:
                        pp.runpp(net2, trafo_model="pi")
                    except pp.LoadflowNotConverged:
                        pass
                is_eq = pp.nets_equal(net1, net2, check_only_results=True)

            if is_eq:
                eq.append(net_name)
            else:
                not_eq.append(net_name)

        # --- plot voltage result difference compared to matpower results
        if isinstance(matpower_excel_result_folder, str) or isinstance(mat_file_folder, str):

            matpower_excel = os.path.join(matpower_excel_result_folder, f"{net_name}.xlsx")
            mat_file = os.path.join(mat_file_folder, f"{net_name}.mat")

            if os.path.isfile(matpower_excel):
                mpc_bus = pd.read_excel(matpower_excel, sheet_name="bus", header=None)
            elif os.path.isfile(mat_file):
                mpc = scipy.io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
                mpc_bus = pd.DataFrame(mpc["mpc"].bus)
                mpc_bus[0] = mpc_bus[0].astype(int)
                va_close_zero = np.isclose(mpc_bus[8], 0)
                if np.sum(va_close_zero) >= 0.8*len(va_close_zero):
                    continue # obviously, results are not included
            else:
                continue

            mpc_bus.iloc[:, 0] -= 1  # adjust for comparison with 0-based python
            vm_mpc = mpc_bus.set_index(0)[7]
            vm_df = pd.concat([net1.res_bus.vm_pu, vm_mpc], axis=1, keys=["pp", "matpower"])
            vm_diff = vm_df.diff(axis=1).iloc[:,-1]
            fig, axs = plt.subplots(nrows=2, sharex=True)
            vm_df.reset_index(drop=True).plot(ax=axs[0])
            vm_diff.reset_index(drop=True).plot(ax=axs[1])
            plt.tight_layout()
            plt.savefig(os.path.join(home, "desktop", f"{net_name}.png"), format="png")
            # print()

    if len(load_err):
        logger.error(f"These networks could not be loaded from json files:\n{load_err}")
    if len(not_eq):
        logger.error(f"The comparison reveald result differences for these networks:\n{not_eq}")

    return eq, not_eq, load_err


def defaulting_folder_integers(folder):
    return folder if isinstance(folder, str) else \
        os.path.join(home, "Desktop", f"power_system_test_case_jsons_{folder}")


if __name__ == "__main__":
    compare_json_folders(
        1,
        os.path.join(home, "desktop", "v2.8.0"),
        os.path.join(home, "desktop", "matpower_excels"),
        os.path.join(home, "Documents", "GIT", "smeinecke", "smeinecke", "python", "networks", "from_mat_files", "mat_files")
        )
    # compare_json_folders(1, os.path.join(home, "desktop", "v2.14.11"), os.path.join(home, "desktop", "matpower_excels"))
    # compare_json_folders(1, os.path.join(home, "desktop", "develop"), os.path.join(home, "desktop", "matpower_excels"))
