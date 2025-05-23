import pytest
import os
import pandas as pd
from pandapower import pp_dir
from pandapower.run import runpp
from pandapower.create import create_bus, create_transformer, create_transformer3w, create_dcline
from pandapower.networks.create_examples import example_simple
from pandapower.networks.power_system_test_cases import case9
from pandapower.networks.cigre_networks import create_cigre_network_mv
from pandapower.timeseries import DFData
from pandapower.control import ConstControl
from pandapower.grid_equivalents.auxiliary import drop_measurements_and_controllers, \
    _check_network, get_boundary_vp, adaptation_phase_shifter, drop_internal_branch_elements
from pandapower.grid_equivalents.get_equivalent import get_equivalent


def test_drop_internal_branch_elements():
    net = example_simple()
    drop_internal_branch_elements(net, net.bus.index)
    assert not net.line.shape[0]
    assert not net.trafo.shape[0]

    net = example_simple()
    n_trafo = net.trafo.shape[0]
    drop_internal_branch_elements(net, net.bus.index, branch_elements=["line"])
    assert not net.line.shape[0]
    assert net.trafo.shape[0] == n_trafo

    net = example_simple()
    n_trafo = net.trafo.shape[0]
    drop_internal_branch_elements(net, [2, 3, 4, 5])
    assert set(net.line.index) == {0, 2, 3}
    assert set(net.trafo.index) == set()

    net = example_simple()
    n_trafo = net.trafo.shape[0]
    drop_internal_branch_elements(net, [4, 5, 6])
    assert set(net.line.index) == {0}
    assert set(net.trafo.index) == {0}


def test_trafo_phase_shifter():
    net = create_cigre_network_mv(with_der="pv_wind")
    net.trafo.loc[0, 'shift_degree'] = 150
    runpp(net)
    net_eq = get_equivalent(net, "rei", [4, 8], [0],
                                                retain_original_internal_indices=True)
    v, p = get_boundary_vp(net_eq, net_eq.bus_lookups)
    net.res_bus.vm_pu = net.res_bus.vm_pu.values + 1e-3
    net.res_bus.va_degree = net.res_bus.va_degree.values + 1e-3
    adaptation_phase_shifter(net, v, p)
    assert len(net.trafo) == 3


def test_drop_measurements_and_controllers():
    # create measurements
    net = case9()
    runpp(net)
    create_bus(net, net.bus.vn_kv.values[0])
    create_bus(net, net.bus.vn_kv.values[0])
    create_bus(net, net.bus.vn_kv.values[0])
    buses = [1, 2, 5, 6, 7, 9, 10, 11]
    create_transformer(net, 1, 9, "0.4 MVA 10/0.4 kV")
    create_transformer3w (net, 2, 10, 11, "63/25/38 MVA 110/20/10 kV")
    net.measurement.loc[0] = ["mb", "v", "bus", 0, 1.0, 0.01, None]
    net.measurement.loc[1] = ["mb", "v", "bus", 5, 1.0, 0.01, None]
    net.measurement.loc[2] = ["mb", "i", "line", 0, 0.9, 0.01, "to"]
    net.measurement.loc[3] = ["mb", "i", "line", 3, 1.3, 0.01, "from"]
    net.measurement.loc[4] = ["mb", "p", "trafo", 0, 89.3, 0.01, "hv"]
    net.measurement.loc[5] = ["mb", "i", "trafo3w", 0, 23.56, 0.01, "mv"]
    assert len(net.measurement) == 6

    # create controllers
    json_path = os.path.join(pp_dir, "test", "opf", "cigre_timeseries_15min.json")
    time_series = pd.read_json(json_path)
    load_ts = pd.DataFrame(index=time_series.index.tolist(), columns=net.load.index.tolist())
    gen_ts = pd.DataFrame(index=time_series.index.tolist(), columns=net.gen.index.tolist())
    for t in range(96):
        load_ts.loc[t] = net.load.p_mw.values * time_series.at[t, "residential"]
        gen_ts.loc[t] = net.gen.p_mw.values * time_series.at[t, "pv"]

    ConstControl(net, element="load", variable="p_mw", element_index=net.load.index.tolist(),
                 profile_name=net.load.index.tolist(), data_source=DFData(load_ts))
    ConstControl(net, element="gen", variable="p_mw", element_index=net.gen.index.tolist(),
                 profile_name=net.gen.index.tolist(), data_source=DFData(gen_ts))
    for i in net.gen.index:
        ConstControl(net, element="gen", variable="p_mw", element_index=i,
                     profile_name=net.gen.index[i], data_source=DFData(gen_ts))
    for i in net.load.index:
        ConstControl(net, element="load", variable="p_mw", element_index=i,
                     profile_name=net.load.index[i], data_source=DFData(load_ts))

    assert net.controller.object[0].__dict__["element_index"] == [0, 1, 2]
    assert net.controller.object[1].__dict__["element_index"] == [0, 1]
    assert net.controller.object[2].__dict__["element_index"] == 0
    assert net.controller.object[3].__dict__["element_index"] == 1

    drop_measurements_and_controllers(net, buses)
    assert len(net.measurement) == 2
    assert len(net.controller) == 3
    assert net.controller.index.tolist() == [0, 4, 6]
    assert net.controller.object[0].__dict__["element_index"] == [0, 2]
    assert net.controller.object[0].__dict__["matching_params"]["element_index"] == [0, 2]


def test_check_network():
    net = case9()
    net.bus.loc[5, 'in_service'] = False
    runpp(net)
    _check_network(net)

    net.bus.loc[5, 'in_service'] = True
    runpp(net)
    create_bus(net, net.bus.vn_kv.values[0])
    create_bus(net, net.bus.vn_kv.values[0])
    create_dcline(net, from_bus=4, to_bus=9, p_mw=1e4, loss_percent=1.2, loss_mw=25, \
                     vm_from_pu=1.01, vm_to_pu=1.02)
    create_dcline(net, from_bus=8, to_bus=10, p_mw=1e4, loss_percent=1.2, loss_mw=25, \
                     vm_from_pu=1.01, vm_to_pu=1.02)
    _check_network(net)
    assert len(net.gen) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])