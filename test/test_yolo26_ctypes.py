# Copyright 2026 Thornbots
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check the C ABI using the same pose fixture as the C++ decoder test."""
import ctypes
import sys

lib = ctypes.CDLL(sys.argv[1])
decode = lib.yolo26_decode_pose
decode.restype = ctypes.c_int64
decode.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int64,
                   ctypes.c_int64, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                   ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
row = [100, 200, 140, 220, 0.8, 2, 100, 200, 0.9, 140, 200, 0.8,
       140, 220, 0.7, 100, 220, 0.6]
tensor = (ctypes.c_float * 36)(*(row + row[:4] + [0.05, 1] + row[6:]))
out = (ctypes.c_float * 36)()
error = ctypes.create_string_buffer(256)
count = decode(tensor, 36, 2, 18, 0.25, out, 2, error, len(error))
assert count == 1, error.value
expected = [120, 210, 40, 20, 0.8, 2] + row[6:]
assert all(abs(a - b) < 1e-5 for a, b in zip(out, expected))
assert decode(tensor, 36, 2, 17, 0.25, out, 2, error, len(error)) == -1
