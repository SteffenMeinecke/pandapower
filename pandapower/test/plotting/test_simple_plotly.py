# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from os.path import join
from tempfile import gettempdir

import pytest

from pandapower.networks.create_examples import example_multivoltage
from pandapower.networks.mv_oberrhein import mv_oberrhein
from pandapower.plotting import create_weighted_marker_trace
from pandapower.plotting.plotly import simple_plotly

try:
    import plotly

    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


@pytest.mark.slow
@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly functions require the plotly package")
def test_simple_plotly_coordinates():
    net = mv_oberrhein(include_substations=True)
    net.load.scaling, net.sgen.scaling = 1, 1
    # different markers and sizemodes as examples
    markers_load = create_weighted_marker_trace(net, elm_type="load", color="red",
                                                patch_type="triangle-up", sizemode="area",
                                                marker_scaling=100)
    markers_sgen = create_weighted_marker_trace(net, elm_type="sgen", color="green",
                                                patch_type="circle-open", sizemode="diameter",
                                                marker_scaling=100, scale_marker_size=0.5)
    fig = simple_plotly(net, filename=join(gettempdir(), "temp-plot.html"), auto_open=False,
                        additional_traces=[markers_sgen, markers_load])
    assert len(fig.data) == (len(net.line) + 1) + (len(net.trafo) + 1) + 6
    # +1 for the infofunc traces,
    # +6 = 1 bus trace + 1 ext_grid trace + 2 weighted marker traces + 2 scale traces


@pytest.mark.slow
@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly functions require the plotly package")
def test_simple_plotly_3w():
    # net with 3W-transformer
    net = example_multivoltage()
    fig = simple_plotly(net, filename=join(gettempdir(), "temp-plot.html"), auto_open=False)
    assert len(fig.data) == (len(net.line) + 1) + (len(net.trafo) + 1) + (len(net.trafo3w) * 3 + 1) + 2
    # +1 is for infofunc traces, +2 = 1 bus trace + 1 ext_grid trace


@pytest.mark.slow
@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly functions require the plotly package")
def test_simple_plotly_no_html():
    net = example_multivoltage()
    # fig without generating a HTML
    fig = simple_plotly(net, filename=None, auto_open=False)
    assert len(fig.data) == (len(net.line) + 1) + (len(net.trafo) + 1) + (len(net.trafo3w)*3 + 1) + 2
    # +1 is for infofunc traces, +2 = 1 bus trace + 1 ext_grid trace


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
