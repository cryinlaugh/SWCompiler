/*************************************************************************
    > File Name: CodeWriter.h
    > Author: wayne
    > Mail:
    > Created Time: 日  4/28 13:07:06 2019
 ************************************************************************/

#include <sstream>
class CodeWriter {
  public:
    CodeWriter() : _indent(0), _new_line(true) {}

    std::string get_code() { return _ss.str(); }

    void indentInc() { _indent++; }
    void indentDec() { _indent--; }

    template <typename T>
    friend CodeWriter &operator<<(CodeWriter &out, const T &obj) {
        //为了处理缩进等问题，不宜直接使用_ss
        std::stringstream tmp_ss;
        tmp_ss << obj;

        for (char c : tmp_ss.str()) {
            if (c == '\n') {
                out._new_line = true;
            } else {
                if (out._new_line) {
                    for (int i = 0; i < out._indent; i++) {
                        out._ss << "    ";
                    }
                    out._new_line = false;
                }
            } // else
            out._ss << c;
        } // for
        return out;
    } //<<

  private:
    std::stringstream _ss;
    int _indent;
    bool _new_line;
};
