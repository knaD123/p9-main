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
        a0_SRI.pppg[0] <--> { delay = 678ms; datarate = 1000000kbps; } <--> a1_USCB.pppg[0];
        a0_SRI.pppg[1] <--> { delay = 869ms; datarate = 1000000kbps; } <--> a2_UCLA.pppg[0];
        a0_SRI.pppg[2] <--> { delay = 1604ms; datarate = 1000000kbps; } <--> a3_UTAH.pppg[0];
        a1_USCB.pppg[1] <--> { delay = 237ms; datarate = 1000000kbps; } <--> a2_UCLA.pppg[1];
        a1_USCB.pppg[2] <--> { delay = 0ms; datarate = 100kbps; } <--> host1.pppg[0];
        a0_SRI.pppg[3] <--> { delay = 0ms; datarate = 100kbps; } <--> target1.pppg[0];
        a1_USCB.pppg[3] <--> { delay = 0ms; datarate = 100kbps; } <--> host2.pppg[0];
        a2_UCLA.pppg[2] <--> { delay = 0ms; datarate = 100kbps; } <--> target2.pppg[0];
        a1_USCB.pppg[4] <--> { delay = 0ms; datarate = 100kbps; } <--> host3.pppg[0];
        a3_UTAH.pppg[1] <--> { delay = 0ms; datarate = 100kbps; } <--> target3.pppg[0];
        a2_UCLA.pppg[3] <--> { delay = 0ms; datarate = 100kbps; } <--> host4.pppg[0];
        a0_SRI.pppg[4] <--> { delay = 0ms; datarate = 100kbps; } <--> target4.pppg[0];
        a2_UCLA.pppg[4] <--> { delay = 0ms; datarate = 100kbps; } <--> host5.pppg[0];
        a1_USCB.pppg[5] <--> { delay = 0ms; datarate = 100kbps; } <--> target5.pppg[0];
        a2_UCLA.pppg[5] <--> { delay = 0ms; datarate = 100kbps; } <--> host6.pppg[0];
        a3_UTAH.pppg[2] <--> { delay = 0ms; datarate = 100kbps; } <--> target6.pppg[0];
        a3_UTAH.pppg[3] <--> { delay = 0ms; datarate = 100kbps; } <--> host7.pppg[0];
        a0_SRI.pppg[5] <--> { delay = 0ms; datarate = 100kbps; } <--> target7.pppg[0];
        a3_UTAH.pppg[4] <--> { delay = 0ms; datarate = 100kbps; } <--> host8.pppg[0];
        a2_UCLA.pppg[6] <--> { delay = 0ms; datarate = 100kbps; } <--> target8.pppg[0];
        a3_UTAH.pppg[5] <--> { delay = 0ms; datarate = 100kbps; } <--> host9.pppg[0];
        a1_USCB.pppg[6] <--> { delay = 0ms; datarate = 100kbps; } <--> target9.pppg[0];
        a0_SRI.pppg[6] <--> { delay = 0ms; datarate = 100kbps; } <--> host10.pppg[0];
        a2_UCLA.pppg[7] <--> { delay = 0ms; datarate = 100kbps; } <--> target10.pppg[0];
        a0_SRI.pppg[7] <--> { delay = 0ms; datarate = 100kbps; } <--> host11.pppg[0];
        a1_USCB.pppg[7] <--> { delay = 0ms; datarate = 100kbps; } <--> target11.pppg[0];
        a0_SRI.pppg[8] <--> { delay = 0ms; datarate = 100kbps; } <--> host12.pppg[0];
        a3_UTAH.pppg[6] <--> { delay = 0ms; datarate = 100kbps; } <--> target12.pppg[0];
}
