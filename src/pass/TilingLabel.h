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

#include <string>
#include <vector>
#include <algorithm>
#include "../graphIR/TensorNode.h"
#include "Label.h"
#include "TileHint.h"
namespace swc {


class TensorTilingLabel: public Label {
private:
    //Label* _label;
    int _tilenum;//
    std::vector<int> _tiles;
    //get tilenum;
    //std::vector<TensorShape*> _slices;
    //std::vector<TileHint> _tilehints;

public:
    TensorTilingLabel(int ndim): _tilenum(ndim) {

        _tiles.reserve(ndim);
        _tilenum = 1;


    };
    ~TensorTilingLabel() {};

    void addTileBydim(int dim, int spiltnum) {
        //simple implement
        // don't allow tile by same dim

        if(_tiles[dim] == 0)
            _tiles[dim] = spiltnum;
        else

            _tiles[dim] *= spiltnum;
        //recalculate tilenum
        _tilenum = _tilenum * spiltnum;


    }

    void replicate(int num) {
        _tilenum = _tilenum * num;
    }

    int getTotalTileNum() {
        return _tilenum;
    }

    int getTileNumByDim(int dim) {
        return _tiles[dim];
    }


};

class OpTilingLabel: public Label {

private:
    //Label* _label;
    std::string _pattern;
    int _replicatenum;
    // int num;
    // string pattern : simple num, map-n-reduce , map-n-without reduce ,filter -n ,scan-n

public:
    OpTilingLabel() {};

    ~OpTilingLabel() {};

    void setReplicateNum(int num) {
        _replicatenum = num;
    }


    int getReplicateNum() {
        return _replicatenum;
    }




    void setPattern(std::string pattern) {
        _pattern = pattern;

    }


    std::string getPattern() {
        return _pattern;
    }

    //map-n-without-reduce

};

}

#endif
