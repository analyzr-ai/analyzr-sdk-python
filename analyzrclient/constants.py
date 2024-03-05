"""
Copyright (c) 2024 Go2Market Insights, Inc
All rights reserved.
https://analyzr.ai

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
from pathlib import Path

#==============================================================================#
#                          GENERAL PARAMETERS                                  #
#==============================================================================#
VERBOSE = True
CLIENT_VERSION = '1.3.72'
HOME = str(Path.home())
TEMP_DIR = '{}/.analyzr'.format(HOME)

#==============================================================================#
#                          REGRESSION PARAMETERS                               #
#==============================================================================#
REGRESSION_DEFAULT_ALGO = 'linear-regression'
