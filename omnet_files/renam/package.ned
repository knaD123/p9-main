package inet.examples.mpls.frrtest;
import inet.common.scenario.ScenarioManager;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.StandardHost;
import inet.node.mpls.MplsRouter;

network renam
{
    submodules:
        configurator: Ipv4NetworkConfigurator;
        a0_Chisinau: MplsRouter {
            parameters:
                peers = "ppp0 ppp1 ppp2 ppp3";
            gates:
                pppg[12];
        }
        a1_Balti: MplsRouter {
            parameters:
                peers = "ppp0";
            gates:
                pppg[9];
        }
        a2________Cahul: MplsRouter {
            parameters:
                peers = "ppp0";
            gates:
                pppg[9];
        }
        a3_StarNet_ISP: MplsRouter {
            parameters:
                peers = "ppp0";
            gates:
                pppg[9];
        }
        a4_RoEduNet: MplsRouter {
            parameters:
                peers = "ppp0";
            gates:
                pppg[9];
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
        host13: StandardHost {
            gates:
                pppg[1];
        }
        target13: StandardHost {
            gates:
                pppg[1];
        }
        host14: StandardHost {
            gates:
                pppg[1];
        }
        target14: StandardHost {
            gates:
                pppg[1];
        }
        host15: StandardHost {
            gates:
                pppg[1];
        }
        target15: StandardHost {
            gates:
                pppg[1];
        }
        host16: StandardHost {
            gates:
                pppg[1];
        }
        target16: StandardHost {
            gates:
                pppg[1];
        }
        host17: StandardHost {
            gates:
                pppg[1];
        }
        target17: StandardHost {
            gates:
                pppg[1];
        }
        host18: StandardHost {
            gates:
                pppg[1];
        }
        target18: StandardHost {
            gates:
                pppg[1];
        }
        host19: StandardHost {
            gates:
                pppg[1];
        }
        target19: StandardHost {
            gates:
                pppg[1];
        }
        host20: StandardHost {
            gates:
                pppg[1];
        }
        target20: StandardHost {
            gates:
                pppg[1];
        }
	connections:
        a0_Chisinau.pppg[0] <--> { delay = 10ms; datarate = 1048576kbps; } <--> a1_Balti.pppg[0];
        a0_Chisinau.pppg[1] <--> { delay = 10ms; datarate = 1048576kbps; } <--> a2________Cahul.pppg[0];
        a0_Chisinau.pppg[2] <--> { delay = 10ms; datarate = 1048576kbps; } <--> a3_StarNet_ISP.pppg[0];
        a0_Chisinau.pppg[3] <--> { delay = 10ms; datarate = 1048576kbps; } <--> a4_RoEduNet.pppg[0];
        a0_Chisinau.pppg[4] <--> { delay = 10ms; datarate = 600kbps; } <--> host1.pppg[0];
        a1_Balti.pppg[1] <--> { delay = 10ms; datarate = 600kbps; } <--> target1.pppg[0];
        a0_Chisinau.pppg[5] <--> { delay = 10ms; datarate = 600kbps; } <--> host2.pppg[0];
        a2________Cahul.pppg[1] <--> { delay = 10ms; datarate = 600kbps; } <--> target2.pppg[0];
        a0_Chisinau.pppg[6] <--> { delay = 10ms; datarate = 600kbps; } <--> host3.pppg[0];
        a3_StarNet_ISP.pppg[1] <--> { delay = 10ms; datarate = 600kbps; } <--> target3.pppg[0];
        a0_Chisinau.pppg[7] <--> { delay = 10ms; datarate = 600kbps; } <--> host4.pppg[0];
        a4_RoEduNet.pppg[1] <--> { delay = 10ms; datarate = 600kbps; } <--> target4.pppg[0];
        a1_Balti.pppg[2] <--> { delay = 10ms; datarate = 600kbps; } <--> host5.pppg[0];
        a0_Chisinau.pppg[8] <--> { delay = 10ms; datarate = 600kbps; } <--> target5.pppg[0];
        a1_Balti.pppg[3] <--> { delay = 10ms; datarate = 600kbps; } <--> host6.pppg[0];
        a2________Cahul.pppg[2] <--> { delay = 10ms; datarate = 600kbps; } <--> target6.pppg[0];
        a1_Balti.pppg[4] <--> { delay = 10ms; datarate = 600kbps; } <--> host7.pppg[0];
        a3_StarNet_ISP.pppg[2] <--> { delay = 10ms; datarate = 600kbps; } <--> target7.pppg[0];
        a1_Balti.pppg[5] <--> { delay = 10ms; datarate = 600kbps; } <--> host8.pppg[0];
        a4_RoEduNet.pppg[2] <--> { delay = 10ms; datarate = 600kbps; } <--> target8.pppg[0];
        a2________Cahul.pppg[3] <--> { delay = 10ms; datarate = 600kbps; } <--> host9.pppg[0];
        a0_Chisinau.pppg[9] <--> { delay = 10ms; datarate = 600kbps; } <--> target9.pppg[0];
        a2________Cahul.pppg[4] <--> { delay = 10ms; datarate = 600kbps; } <--> host10.pppg[0];
        a1_Balti.pppg[6] <--> { delay = 10ms; datarate = 600kbps; } <--> target10.pppg[0];
        a2________Cahul.pppg[5] <--> { delay = 10ms; datarate = 600kbps; } <--> host11.pppg[0];
        a3_StarNet_ISP.pppg[3] <--> { delay = 10ms; datarate = 600kbps; } <--> target11.pppg[0];
        a2________Cahul.pppg[6] <--> { delay = 10ms; datarate = 600kbps; } <--> host12.pppg[0];
        a4_RoEduNet.pppg[3] <--> { delay = 10ms; datarate = 600kbps; } <--> target12.pppg[0];
        a3_StarNet_ISP.pppg[4] <--> { delay = 10ms; datarate = 600kbps; } <--> host13.pppg[0];
        a0_Chisinau.pppg[10] <--> { delay = 10ms; datarate = 600kbps; } <--> target13.pppg[0];
        a3_StarNet_ISP.pppg[5] <--> { delay = 10ms; datarate = 600kbps; } <--> host14.pppg[0];
        a1_Balti.pppg[7] <--> { delay = 10ms; datarate = 600kbps; } <--> target14.pppg[0];
        a3_StarNet_ISP.pppg[6] <--> { delay = 10ms; datarate = 600kbps; } <--> host15.pppg[0];
        a2________Cahul.pppg[7] <--> { delay = 10ms; datarate = 600kbps; } <--> target15.pppg[0];
        a3_StarNet_ISP.pppg[7] <--> { delay = 10ms; datarate = 600kbps; } <--> host16.pppg[0];
        a4_RoEduNet.pppg[4] <--> { delay = 10ms; datarate = 600kbps; } <--> target16.pppg[0];
        a4_RoEduNet.pppg[5] <--> { delay = 10ms; datarate = 600kbps; } <--> host17.pppg[0];
        a0_Chisinau.pppg[11] <--> { delay = 10ms; datarate = 600kbps; } <--> target17.pppg[0];
        a4_RoEduNet.pppg[6] <--> { delay = 10ms; datarate = 600kbps; } <--> host18.pppg[0];
        a1_Balti.pppg[8] <--> { delay = 10ms; datarate = 600kbps; } <--> target18.pppg[0];
        a4_RoEduNet.pppg[7] <--> { delay = 10ms; datarate = 600kbps; } <--> host19.pppg[0];
        a2________Cahul.pppg[8] <--> { delay = 10ms; datarate = 600kbps; } <--> target19.pppg[0];
        a4_RoEduNet.pppg[8] <--> { delay = 10ms; datarate = 600kbps; } <--> host20.pppg[0];
        a3_StarNet_ISP.pppg[8] <--> { delay = 10ms; datarate = 600kbps; } <--> target20.pppg[0];
}
