import os
import pathlib
from io import StringIO
import numpy as np
import pandas as pd

from pandapower.converter.matpower.from_mpc import _mpc2ppc
from pandapower.converter.pypower.from_ppc import from_ppc
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.run import runpp
import pandapower.networks as pn
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
home = str(pathlib.Path.home())


def create_power_system_test_case_jsons(output_folder=None):
    """Creates all power system test cases that originates from matpower case files as pandpower
    json files into the output_folder (defaults to a desktop folder)
    """
    if output_folder is None:
        val = 1
        output_folder = os.path.join(home, "Desktop", f"power_system_test_case_jsons_{val}")
        while os.path.exists(val):
            val += 1
            output_folder = os.path.join(home, "Desktop", f"power_system_test_case_jsons_{val}")

    test_case_df, input_data_folders = test_case_table_to_create()

    for row in test_case_df.itertuples():
        logging.info(f"Start loading and converting {row.name}")

        # --- load the net
        ppc = load_ppc(row.name, input_data_folders[row.folder_no])

        # --- correct, convert and adjust the net
        correct_or_adapt_input_data(ppc, net_name)
        net = from_ppc(ppc, f_hz=row.f_hz)
        try_to_take_bus_geo_data_from_existing(net, row.name)
        try_to_provide_pf_results(net, row.name)

        # --- store the net
        pp.to_json(net, os.path.join(output_folder, row.name + '.json'))

    logging.info(f"Loading, converting, and saving of all power system test cases completed.")


def load_ppc(net_name, folder):
    mat_case_path = os.path.join(folder, net_name + '.mat')
    return _mpc2ppc(mat_case_path, casename_mpc_file='mpc')


def test_case_table_to_create():
    table_str = """
name;f_hz;folder_no
case4gs;60;1
case6ww;60;1
case9;60;1
case14;60;1
case24_ieee_rts;60;1
case30;60;1
case39;60;1
case57;60;1
case118;60;1
case300;60;1
case5;60;1
case_ieee30;60;1
case33bw;60;1
case89pegase;50;1
case145;60;1
case_illinois200;60;1
case1354pegase;50;1
case1888rte;50;1
case2848rte;50;1
case2869pegase;50;1
case3120sp;50;1
case6470rte;50;1
case6495rte;50;1
case6515rte;50;1
case9241pegase;50;1
GBnetwork;50;2
GBreducednetwork;50;2
iceland;50;2
"""
    input_data_folders = pd.Series({
        1: os.path.join("Documents", "MATLAB", "matpower8.0"),
        2: os.path.join("Documents", "further_mpc_data"),
    })
    table = pd.read_csv(StringIO(table_str), sep=";")
    return table, input_data_folders


def correct_or_adapt_input_data(ppc, net_name):
    """In the original data of some power system test cases inappropriate data is included. This
    function corrects these at the stage of ppc format, i.e. before conversion by from_ppc().
    """
    if net_name == "case14":
        # fix missing vn_kv values
        ppc["bus"][:, 9] = np.array([135]*5+[0.208, 14, 12]+[0.208]*6)
    elif net_name == "case57":
        # fix missing vn_kv values
        ppc["bus"][:, 9] = np.array([115]*17+[500]*3+[138]*4+[345]+[161]*4+[345]*4+[138]*7 +
                                    [230]*3+[138]*8+[161]*4+[230]*2)
    elif net_name == "case118":
        # fix vn_kv values whereby branches do not lead to i0_percent < 0
        ppc["bus"][67, 9] = 161
        ppc["bus"][115, 9] = 345
    #    ppc["branch"][-4, 4] = 0  # made before june 2018
    #    ppc["branch"][133, 4] = 0  # made before june 2018
    elif net_name == "case1888rte":
        _choose_slack_buses_with_connected_gens(ppc, 1267)
    elif net_name == "case1951rte":
        _choose_slack_buses_with_connected_gens(ppc, 179)
    elif net_name == "case2848rte":
        _choose_slack_buses_with_connected_gens(ppc, 239)
    elif net_name == "case6468rte":
        _choose_slack_buses_with_connected_gens(ppc, 6218)
    elif net_name == "case6470rte":
        _choose_slack_buses_with_connected_gens(ppc, 5988)
    elif net_name == "case6495rte":
        _choose_slack_buses_with_connected_gens(ppc, [6077, 6161, 6307, 6308, 6309, 6310])
    elif net_name == "case6515rte":
        _choose_slack_buses_with_connected_gens(ppc, 6171)


def _choose_slack_buses_with_connected_gens(ppc, new_slack_buses):
    """In the original data of some power system test cases inappropriate buses are selected as
    reference buses, i.e. slack buses. For example, a bus without connected generator is a bad
    assumption and cannot be modeled by pandapower.
    new_slack_buses defines which buses should be the slack buses instead.

    Note
    ----
    new_slack_buses must be the name of the new slack buses
    """
    new_slack_buses = new_slack_buses if hasattr(new_slack_buses, '__iter__') else [new_slack_buses]
    idx_new_slack_buses = [idx for idx in range(len(ppc["bus"])) if ppc["bus"][idx, 0] in
                           new_slack_buses]
    idx_new_slack_gens = [idx for idx in range(len(ppc["gen"])) if ppc["gen"][idx, 0] in
                          new_slack_buses]

    idx_original_slack_buses = np.where(ppc["bus"][:, 1] == 3)[0]
    name_original_slack_buses = ppc["bus"][idx_original_slack_buses, 0][0]
    name_original_slack_buses = name_original_slack_buses if hasattr(
        name_original_slack_buses, '__iter__') else [int(name_original_slack_buses)]
    idx_original_slack_gens = [idx for idx in range(len(ppc["gen"])) if ppc["gen"][idx, 0] in
                               name_original_slack_buses]

    assert len(idx_original_slack_gens) == 0  # there is no gen connected to original slack buses
    assert len(idx_new_slack_gens) >= len(new_slack_buses)  # all new slack buses have connected gens

    # --- change original slack buses to new_slack_buses
    ppc["bus"][idx_original_slack_buses, 1] = 2
    ppc["bus"][idx_new_slack_buses, 1] = 3  # add missing slack bus with gen


def try_to_take_bus_geo_data_from_existing(net, net_name):
    try:
        fnc = getattr(pn, net_name)
        existing_net = fnc()
        net.bus.geo = existing_net.bus.geo
    except AttributeError:
        create_generic_coordinates(net)


def try_to_provide_pf_results(net, net_name):
    try:
        runpp(net, trafo_model="pi")
    except (LoadflowNotConverged, KeyError):
        logger.info(f"Power flow calculation of {row.name} did not converge.")


if __name__ == "__main__":
    create_power_system_test_case_jsons()

    ### check
    # bus.geo
    # max_i_ka, max_loading_percent
    # poly_cost
