# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
import pytest

from pandapower.networks.create_examples import example_simple, example_multivoltage
from pandapower.run import runpp


def test_create_simple():
    net = example_simple()
    runpp(net)
    assert net.converged
    for element in ["bus", "line", "gen", "sgen", "shunt", "trafo", "load", "ext_grid"]:
        assert len(net[element]) >= 1
    for et in ["l", "b"]:
        assert len(net.switch[net.switch.et == et]) >= 1


def test_create_realistic():
    net = example_multivoltage()
    runpp(net)
    assert net.converged
    for element in ["bus", "line", "gen", "sgen", "shunt", "trafo", "trafo3w", "load", "ext_grid",
                    "impedance", "xward"]:
        assert len(net[element]) >= 1
    for et in ["l", "b", "t"]:
        assert len(net.switch[net.switch.et == et]) >= 1
    for type_ in ["CB", "DS", "LBS"]:
        assert len(net.switch[net.switch.type == type_]) >= 1
    assert len(net.switch[net.switch.closed]) >= 1
    assert len(net.switch[~net.switch.closed]) >= 1
    all_vn_kv = pd.Series([380, 110, 20, 10, 0.4])
    assert net.bus.vn_kv.isin(all_vn_kv).all()


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
