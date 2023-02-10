package inet.examples.mpls.arpanet196912;
import inet.common.scenario.ScenarioManager;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.StandardHost;
import inet.node.mpls.MplsRouter;

network arpanet196912
{
    submodules:
        configurator: Ipv4NetworkConfigurator;
        a0_SRI: MplsRouter {
            parameters:
                peers = "ppp0 ppp1 ppp2";
            gates:
                pppg[9];
        }
        a1_USCB: MplsRouter {
            parameters:
                peers = "ppp0 ppp1";
            gates:
                pppg[8];
        }
        a2_UCLA: MplsRouter {
            parameters:
                peers = "ppp0 ppp1";
            gates:
                pppg[8];
        }
        a3_UTAH: MplsRouter {
            parameters:
                peers = "ppp0";
            gates:
                pppg[7];
        }
        host1: StandardHost {
            gates:
                pppg[1];
        }
        target1: StandardHost {
            gates:
                pppg[1];
        }
        host2: StandardHost {
            gates:
                pppg[1];
        }
        target2: StandardHost {
            gates:
                pppg[1];
        }
        host3: StandardHost {
            gates:
                pppg[1];
        }
        target3: StandardHost {
            gates:
                pppg[1];
        }
        host4: StandardHost {
            gates:
                pppg[1];
        }
        target4: StandardHost {
            gates:
                pppg[1];
        }
        host5: StandardHost {
            gates:
                pppg[1];
        }
        target5: StandardHost {
            gates:
                pppg[1];
        }
        host6: StandardHost {
            gates:
                pppg[1];
        }
        target6: StandardHost {
            gates:
                pppg[1];
        }
        host7: StandardHost {
            gates:
                pppg[1];
        }
        target7: StandardHost {
            gates:
                pppg[1];
        }
        host8: StandardHost {
            gates:
                pppg[1];
        }
        target8: StandardHost {
            gates:
                pppg[1];
        }
        host9: StandardHost {
            gates:
                pppg[1];
        }
        target9: StandardHost {
            gates:
                pppg[1];
        }
        host10: StandardHost {
            gates:
                pppg[1];
        }
        target10: StandardHost {
            gates:
                pppg[1];
        }
        host11: StandardHost {
            gates:
                pppg[1];
        }
        target11: StandardHost {
            gates:
                pppg[1];
        }
        host12: StandardHost {
            gates:
                pppg[1];
        }
        target12: StandardHost {
            gates:
                pppg[1];
        }
	connections:
        a0_SRI.pppg[0] <--> { delay = 678ms; datarate = 1000000bps; @statistic[utilization](record=max,timeavg,vector); } <--> a1_USCB.pppg[0];
        a0_SRI.pppg[1] <--> { delay = 869ms; datarate = 1000000bps; @statistic[utilization](record=max,timeavg,vector); } <--> a2_UCLA.pppg[0];
        a0_SRI.pppg[2] <--> { delay = 1604ms; datarate = 1000000bps; @statistic[utilization](record=max,timeavg,vector); } <--> a3_UTAH.pppg[0];
        a1_USCB.pppg[1] <--> { delay = 237ms; datarate = 1000000bps; @statistic[utilization](record=max,timeavg,vector); } <--> a2_UCLA.pppg[1];
        a1_USCB.pppg[2] <--> { delay = 0ms; datarate = 100Gbps; } <--> host1.pppg[0];
        a0_SRI.pppg[3] <--> { delay = 0ms; datarate = 100Gbps; } <--> target1.pppg[0];
        a1_USCB.pppg[3] <--> { delay = 0ms; datarate = 100Gbps; } <--> host2.pppg[0];
        a2_UCLA.pppg[2] <--> { delay = 0ms; datarate = 100Gbps; } <--> target2.pppg[0];
        a1_USCB.pppg[4] <--> { delay = 0ms; datarate = 100Gbps; } <--> host3.pppg[0];
        a3_UTAH.pppg[1] <--> { delay = 0ms; datarate = 100Gbps; } <--> target3.pppg[0];
        a2_UCLA.pppg[3] <--> { delay = 0ms; datarate = 100Gbps; } <--> host4.pppg[0];
        a0_SRI.pppg[4] <--> { delay = 0ms; datarate = 100Gbps; } <--> target4.pppg[0];
        a2_UCLA.pppg[4] <--> { delay = 0ms; datarate = 100Gbps; } <--> host5.pppg[0];
        a1_USCB.pppg[5] <--> { delay = 0ms; datarate = 100Gbps; } <--> target5.pppg[0];
        a2_UCLA.pppg[5] <--> { delay = 0ms; datarate = 100Gbps; } <--> host6.pppg[0];
        a3_UTAH.pppg[2] <--> { delay = 0ms; datarate = 100Gbps; } <--> target6.pppg[0];
        a3_UTAH.pppg[3] <--> { delay = 0ms; datarate = 100Gbps; } <--> host7.pppg[0];
        a0_SRI.pppg[5] <--> { delay = 0ms; datarate = 100Gbps; } <--> target7.pppg[0];
        a3_UTAH.pppg[4] <--> { delay = 0ms; datarate = 100Gbps; } <--> host8.pppg[0];
        a2_UCLA.pppg[6] <--> { delay = 0ms; datarate = 100Gbps; } <--> target8.pppg[0];
        a3_UTAH.pppg[5] <--> { delay = 0ms; datarate = 100Gbps; } <--> host9.pppg[0];
        a1_USCB.pppg[6] <--> { delay = 0ms; datarate = 100Gbps; } <--> target9.pppg[0];
        a0_SRI.pppg[6] <--> { delay = 0ms; datarate = 100Gbps; } <--> host10.pppg[0];
        a2_UCLA.pppg[7] <--> { delay = 0ms; datarate = 100Gbps; } <--> target10.pppg[0];
        a0_SRI.pppg[7] <--> { delay = 0ms; datarate = 100Gbps; } <--> host11.pppg[0];
        a1_USCB.pppg[7] <--> { delay = 0ms; datarate = 100Gbps; } <--> target11.pppg[0];
        a0_SRI.pppg[8] <--> { delay = 0ms; datarate = 100Gbps; } <--> host12.pppg[0];
        a3_UTAH.pppg[6] <--> { delay = 0ms; datarate = 100Gbps; } <--> target12.pppg[0];
}
