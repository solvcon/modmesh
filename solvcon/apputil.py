# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


"""
Tools to run applications
"""


# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


import importlib
import rlcompleter


__all__ = [
    'environ',
    'AppEnvironment',
    'get_current_appenv',
    'get_completions',
    'run_code',
    'stop_code',
]


# All environment objects of this process.
environ = {}


class AppEnvironment:
    """
    Collects the environment for an application.

    :ivar globals:
        The global namespace of the application.
    :ivar locals:
        The local namespace of the application.
    """
    def __init__(self, name):
        self.globals = {
            # Give the application an alias of the top package.
            'sc': importlib.import_module('solvcon'),
            'appenv': self,
        }
        self.locals = {}
        self.name = name
        # Each run of the application appends a new environment.
        environ[name] = self

    def run_code(self, code):
        exec(code, self.globals, self.locals)


def get_appenv(name=None):
    if None is name:
        for i in range(10):
            name = f'anonymous{i}'
            if name not in environ:
                break
        else:
            raise ValueError("hit limit of anonymous environments (10)")
    app = environ.get(name, None)
    if None is app:
        app = AppEnvironment(name)
    return app


get_appenv(name='master')


def get_current_appenv():
    has_key = False
    for k in reversed(environ):
        has_key = True
        break
    if not has_key:
        raise KeyError("No AppEnviron is available")
    return environ[k]


def get_completions(text):
    aenv = get_current_appenv()
    namespace = {'__builtins__': __builtins__}
    namespace.update(aenv.globals)
    namespace.update(aenv.locals)
    completer = rlcompleter.Completer(namespace)
    completions = []
    i = 0
    while True:
        c = completer.complete(text, i)
        if c is None:
            break
        completions.append(c)
        i += 1
    return completions


def run_code(code):
    aenv = get_current_appenv()
    aenv.run_code(code)


def stop_code(appenvobj=None):
    if None is appenvobj:
        environ.clear()
    else:
        indices = [i for i, o in enumerate(environ) if o == appenvobj]
        indices = reversed(indices)
        for i in indices:
            del environ[i]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
