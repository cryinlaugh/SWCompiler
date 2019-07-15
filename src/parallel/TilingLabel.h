/***********************************************
#
#      Filename: TilingLable.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-01-21 17:54:42
# Last Modified: 2019-01-21 17:54:42
***********************************************/

#ifndef _TILELABEL_H
#define _TILELABEL_H

#include "../graphIR/TensorNode.h"
#include "pass/Label.h"
#include <algorithm>
#include <string>
#include <vector>
namespace swc {

class TensorTilingLabel : public Label {
private:
    int _tilenum; //
    std::vector<int> _tiles;
public:
    TensorTilingLabel(int ndim) : Label(){
        _tiles.reserve(ndim);
        _tilenum = 1;
    };
    ~TensorTilingLabel() {};

    void addTileBydim(int dim, int spiltnum) {
        // simple implement
        // don't allow tile by same dim



        //_tiles[dim] *= spiltnum;
        // recalculate tilenum
        _tilenum = _tilenum * spiltnum;
    }

    void replicate(int num) {
        _tilenum = _tilenum * num;
    }

    void reduce(int num ) {
        _tilenum = _tilenum * num;
    }
    int getTotalTileNum() {
        return _tilenum;
    }

    int getTileNumByDim(int dim) {
        return _tiles[dim];
    }

};

class OpTilingLabel : public Label {

private:
    // Label* _label;
    int  _type; // withreduce or without reduce
    int _replicatenum;//pattern : simple num, map-n-reduce , map-n-without reduce ,filter

public:
    OpTilingLabel() : Label() {};

    ~OpTilingLabel() {};

    void setReplicateNum(int num) {
        _replicatenum = num;
    }

    int getReplicateNum() {
        return _replicatenum;
    }

    void setPattern(int type) {
        _type = type;
    }

    int getPattern() {
        return _type;
    }

    // map-n-without-reduce
};

} // namespace swc

#endif
