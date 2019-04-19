/*************************************************************************
    > File Name: AutoDiff.h
    > Author: wayne
    > Mail:
    > Created Time: 三  4/10 17:25:58 2019
 ************************************************************************/

struct TrainingProfile;

namespace swc {
class IRGraph;

IRGraph *getTrainNet(IRGraph *graph, TrainingProfile &profile);

} // namespace swc
