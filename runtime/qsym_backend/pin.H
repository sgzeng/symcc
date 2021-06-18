// This file is part of SymCC.
//
// SymCC is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// SymCC is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// SymCC. If not, see <https://www.gnu.org/licenses/>.

#ifndef PIN_H
#define PIN_H

//
// This file provides mocks for everything that would normally be included from
// Pin, so that we don't have to change all the Qsym code.
//

#include <cstdint>
#include <string>
#include <sstream>
#include <random> 
#include <iostream> 
#include <fstream>
#include <map>
#include <vector>
#include <hiredis/hiredis.h>

using namespace std;

typedef unsigned int USIZE;
typedef unsigned int UINT;
typedef int INT;
typedef int32_t INT32;
typedef uint64_t __uint64;
typedef uintptr_t ADDRINT;
typedef uint8_t  UINT8;             ///< 8-bit unsigned integer
typedef uint16_t UINT16;            ///< 16-bit unsigned integer
typedef uint32_t UINT32;            ///< 32-bit unsigned integer
typedef uint64_t UINT64;            ///< 64-bit unsigned integer
typedef int8_t  INT8;               ///< 8-bit signed integer
typedef int16_t INT16;              ///< 16-bit signed integer
typedef int32_t INT32;              ///< 32-bit signed integer
typedef int64_t INT64;              ///< 64-bit signed integer

/// Return the hex representation of a number.
template<typename T> string hexstr(T arg) {
  stringstream stream;
  stream << hex << arg;
  return stream.str();
}

#define decstr(x) to_string(x)

#endif
