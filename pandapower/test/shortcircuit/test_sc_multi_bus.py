# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.create import create_empty_network, create_bus, create_line, create_ext_grid, create_transformer, \
    create_sgen
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.test.shortcircuit.test_meshing_detection import meshed_grid


@pytest.fixture
def radial_grid():
    net = create_empty_network(sn_mva=2.)
    b0 = create_bus(net, 220)
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    create_ext_grid(net, b0, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    create_transformer(net, b0, b1, "100 MVA 220/110 kV")
    create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=20.)
    create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", length_km=15.)
    return net


@pytest.fixture
def three_bus_big_sgen_example():
    net = create_empty_network(sn_mva=3)
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)

    create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=20.)
    create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", length_km=15.)
    net.line["endtemp_degree"] = 80

    create_sgen(net, b2, sn_mva=200., p_mw=0, k=1.2)
    return net


def test_radial_network(radial_grid):
    net = radial_grid
    sc_bus = 3
    calc_sc(net)
    ik = net.res_bus_sc.ikss_ka.at[sc_bus]
    calc_sc(net, bus=sc_bus, inverse_y=False, branch_results=True, return_all_currents=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[sc_bus], ik)
    assert np.isclose(net.res_line_sc.ikss_ka.loc[(1, sc_bus)], ik)
    assert np.isclose(net.res_line_sc.ikss_ka.loc[(0, sc_bus)], ik)
    assert np.isclose(net.res_trafo_sc.ikss_lv_ka.loc[(0, sc_bus)], ik)
    trafo_ratio = net.trafo.vn_lv_kv.values / net.trafo.vn_hv_kv.values
    assert np.isclose(net.res_trafo_sc.ikss_hv_ka.loc[(0, sc_bus)], ik * trafo_ratio)

    sc_bus = 2
    calc_sc(net)
    ik = net.res_bus_sc.ikss_ka.at[sc_bus]
    calc_sc(net, bus=sc_bus, inverse_y=False, branch_results=True, return_all_currents=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[sc_bus], ik)
    assert np.isclose(net.res_line_sc.ikss_ka.loc[(1, sc_bus)], 0)
    assert np.isclose(net.res_line_sc.ikss_ka.loc[(0, sc_bus)], ik)
    assert np.isclose(net.res_trafo_sc.ikss_lv_ka.loc[(0, sc_bus)], ik)
    trafo_ratio = net.trafo.vn_lv_kv.values / net.trafo.vn_hv_kv.values
    assert np.isclose(net.res_trafo_sc.ikss_hv_ka.loc[(0, sc_bus)], ik * trafo_ratio)


def test_meshed_network(meshed_grid):
    net = meshed_grid
    calc_sc(net)
    sc_bus = 5
    ik = net.res_bus_sc.ikss_ka.at[sc_bus]

    calc_sc(net, bus=sc_bus, inverse_y=False, branch_results=True, return_all_currents=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[sc_bus], ik)
    line_ix = net.line[(net.line.to_bus == sc_bus) | (net.line.from_bus == sc_bus)].index
    line_flow_into_sc = net.res_line_sc.loc[(line_ix, sc_bus), "ikss_ka"].sum()
    assert np.isclose(line_flow_into_sc, ik, atol=2e-3)


def test_big_gen_network(three_bus_big_sgen_example):
    net = three_bus_big_sgen_example
    sc_bus = [0, 1, 2]
    calc_sc(net, bus=sc_bus, branch_results=True, return_all_currents=True, inverse_y=False)
    assert np.isclose(net.res_line_sc.loc[(0, 0), "ikss_ka"], 1.25967331, atol=1e-3)
    assert np.isclose(net.res_line_sc.loc[(1, 0), "ikss_ka"], 0., atol=2e-3)
    assert np.isclose(net.res_line_sc.loc[(0, 2), "ikss_ka"], 0.46221808, atol=1e-3)
    assert np.isclose(net.res_line_sc.loc[(1, 2), "ikss_ka"], 1.72233192, atol=1e-3)


def test_big_gen_network_v2(three_bus_big_sgen_example):
    net = three_bus_big_sgen_example
    sc_bus = [0, 2]
    calc_sc(net, bus=sc_bus, branch_results=True, return_all_currents=True, inverse_y=False)
    assert np.isclose(net.res_line_sc.loc[(0, 0), "ikss_ka"], 1.25967331, atol=1e-3)
    assert np.isclose(net.res_line_sc.loc[(1, 0), "ikss_ka"], 0., atol=2e-3)
    assert np.isclose(net.res_line_sc.loc[(0, 2), "ikss_ka"], 0.46221808, atol=1e-3)
    assert np.isclose(net.res_line_sc.loc[(1, 2), "ikss_ka"], 1.72233192, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
