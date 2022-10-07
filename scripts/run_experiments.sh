

./scripts/setup-venv.sh #Sets up virtual env and install requirements

           # Downloads topologies (Currently already in folder)

FILTER="$1" #Filter for which

./scripts/create_confs.sh no "${FILTER}" # Create configurations that will be run. However, this will take a long time. To only create a subset of the problems, filter to only those topologies starting with `zoo_A` by `./scripts/create_confs.sh no zoo_A`.


#Which algorithms that should be run
declare -a CONFS_TO_RUN=("rmpls" "gft" "rsvp-fn" "tba-simple" "inout-disjoint_max-mem=2" "inout-disjoint_max-mem=3" "inout-disjoint_max-mem=4" "inout-disjoint_max-mem=5" "inout-disjoint_max-mem=6" "inout-disjoint_max-mem=7" "inout-disjoint_max-mem=8" "inout-disjoint_max-mem=9" "inout-disjoint_max-mem=10" "inout-disjoint_max-mem=11" "inout-disjoint_max-mem=12" "inout-disjoint_max-mem=13" "inout-disjoint_max-mem=14" "inout-disjoint_max-mem=15" "inout-disjoint_max-mem=16" "inout-disjoint_max-mem=17" "inout-disjoint_max-mem=18" "inout-disjoint_max-mem=19" "inout-disjoint_max-mem=20" "inout-disjoint_max-mem=21" "inout-disjoint_max-mem=22" "inout-disjoint_max-mem=23" "inout-disjoint_max-mem=24" "inout-disjoint_max-mem=25" "tba-complex_max-mem=2" "tba-complex_max-mem=3" "tba-complex_max-mem=4" "tba-complex_max-mem=5" "tba-complex_max-mem=6" "tba-complex_max-mem=7" "tba-complex_max-mem=8" "tba-complex_max-mem=9" "tba-complex_max-mem=10" "tba-complex_max-mem=11" "tba-complex_max-mem=12" "tba-complex_max-mem=13" "tba-complex_max-mem=14" "tba-complex_max-mem=15" "tba-complex_max-mem=16" "tba-complex_max-mem=17" "tba-complex_max-mem=18" "tba-complex_max-mem=19" "tba-complex_max-mem=20" "tba-complex_max-mem=21" "tba-complex_max-mem=22" "tba-complex_max-mem=23" "tba-complex_max-mem=24" "tba-complex_max-mem=25")

if [ "$2" != "all" ]
then
    CONFS_TO_RUN=("${@:2}")
fi

#Possible algorithms are: 
# [Conf-name]                     |     [Name in paper]
# inout-disjoint_max-mem=X                  FBR
# tba-complex_max-mem=X                     E-CA
# rmpls                                     R-MPLS
# gft                                       GFT-CA
# rsvp-fn                                   RSVP-FN
# tba-simple                                B-CA

#where X is a integer limit on the number of rules per router per demand.


for CONF in "${CONFS_TO_RUN[@]}"
do
    ./scripts/run-problems-subset.sh ${CONF} 300 ${FILTER}                #Simulate the CONFS. Second parameter is a limit on number of topologies to run on.
done

python3 latex_generation_all.py

python3 ./scripts/compile_latex_plots_to_standalone.py latex