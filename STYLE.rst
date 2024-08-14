====
Coding Style Guide
====

A code style guideline is to help developers align how they write and change
code. The consistency reduces the cost to  maintain and develop the code, and
the former matters more than the latter, because the former costs more than the
latter.

modmesh uses `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`__ to
lint C++ code and `flake8 <https://flake8.pycqa.org/>`__ to lint Python code
according to `PEP-8 <https://peps.python.org/pep-0008/>`__. We mind the code
style when adding new code and changing existing code. The rules of thumb are:

1. The linters must be clean. Before creating and updating a
   `pull request <https://docs.github.com/en/pull-requests/>`__, run:

   .. code-block:: bash

     make lint

2. Read the code nearby and follow the style. Start from the functions and
   classes that the code resides in. Then get familiar with the style in the
   file and follow it. Familiar with the code in the module(s) if time permits.
3. Use the style guide.

Indentation and file format
====

Use 4 white spaces for indentation. Do not use a tab.

C++ files do not have a text width limit, but it is good for a line to be less
than 120 characters. Python files should use a text width of 79 characters.

Use UTF-8 as file encoding and `UNIX text file format
<http://en.wikipedia.org/wiki/Newline>`__. Do not use DOS file format.

Vim modelines
----

Even if you do not use vim, add the modeline at the end of files to document
the required file format:

* ``ff=unix``: Use the `UNIX text file format
  <http://en.wikipedia.org/wiki/Newline>`__ (``\n`` line ending).
* ``fenc=utf8``: Use UTF-8 for encoding.
* ``et``: Expand tabs. Do not use tabs for modmesh.
* ``sw=4 ts=4 sts=4``: Use 4 white spaces for tabs.

The modeline for C++ is:

.. code-block:: cpp

  // vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

The modeline for Python is:

.. code-block:: python

  # vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

In Python, set the text width to 79 (``tw=79``).

Space and Blank Line
----

Leave a space behind ``,`` (commas):

.. code-block:: cpp

  void help_something(int32_t serial, double value);

Use a blank line between the definitions of classes and functions.

Naming
====

Do not use a name (especially for a variable) with only 1 character.

Prefer to use ``UPPER_CASE`` for constants. In C++ sometimes ``snake_case``
is preferred when it involves a foreign code base.

Functions and variables use ``snake_case`` and classes use ``CamelCase`` in
both C++ and Python.

Member data and functions in a C++ class use the same naming convention
regardless of access (``public``, ``protected``, and ``private``). Member data
should be prefixed with ``m_`` like ``m_snake_case``, unless it is for a POD
(plain-old-data) struct.

C++ types (classes) for type aliasing and template meta-programming follow STL
and use ``snake_case_t`` or ``snake_case_type``, e.g., ``size_type``

.. code-block:: cpp

  class MyPowerHouse
  {

  public:

      void do_something();

  private:

      void help_something();

      int32_t m_serial_number;

  }; /* end class MyPowerHouse */

  struct PureData
  {

      // Member data names in POD are usually short for easy access.
      int32_t serial;
      double x, y;

  }; /* end struct PureData */

In a Python class, public attributes and methods (member functions) use normal
``snake_case``. Non-public (nothing is really private in Python) attributes and
methods use ``_leading_underscore_snake_case`` (unmangled) and
``__double_leading_underscore_snake_case`` (mangled).

Python exceptions are Python classes and use ``CamelCase``.

Do the best to name a function like ``verb_objective()`` (in both C++ and
Python).

.. code-block:: python

  # function.
  take_some_action(from_this, with_that)
  # method.
  some_object.do_something(with_some_information)

Acronym
----

Treat acronyms like a word. Do not make them all-upper-cases in names.

.. code-block:: cpp

  // "Http" is treated like a word in CamelCase.
  class HttpRequest
  {
      // "http" is treated like a word in snake_case.
      void update_http_header();
  } /* end class HttpRequest */

Qt
----

For Qt sub-classes, follow the Qt naming style, but prefix with ``R`` instead
of ``Q`` and put them in the ``modmesh`` namespace. (Why "``R``"? It is the
next character than "``Q``" and we want to distinguish the classes derived in
modmesh.) Use ``camelCase`` (note the leading lower-case character) for
functions. Member data should use ``m_snake_case`` as other modmesh C++ class.

Iterating Counter
----

Iterating counters start with ``i``, ``j``, ``k``.

- Trivial indexing variables can be named as ``it``, ``jt``, or ``kt``.
- Standalone ``i``, ``j``, and ``k`` should never be used to name a variable
  because they are too short.

Shorthands for Unstructured Meshes
----

Code for the unstructured meshes carries geometrical terms and needs shorthands
to keep the line width reasonable.

- Two-character names for nodes, faces, and cells:

  - ``nd``: node/vertex.
  - ``fc``: face.
  - ``cl``: cell.
- For example, ``icl`` is a counter of cell.
- The following prefices often (but not always) means serial numbers:

  - ``nxx``: number of ``xx``, e.g., ``ncl`` is number of cells.
  - ``mxx``: maximum number of ``xx``, e.g., ``mfc`` is the maximum number of
    faces.

More examples:

- ``clnnd`` means number of nodes belonging to a cell.
- ``FCMND`` means maximum number of nodes for a face.
- ``icl`` means the first-level (iterating) index of cell.
- ``jfc`` means the second-level (iterating) index of face.
- Some special iterators used in code, such as:

  - ``clfcs(icl, ifl)``: get the ``ifl``-th face in ``icl``-th cell.
  - ``fcnds(ifc, inf)``: get the ``inf``-th fact in ``ifc``-th face.

Python import
====

Never import everything ("``import *``" or "``from mod import *``").

Only import modules, like:

.. code-block:: python

  # Mind the order of the lines importing the modules.
  # Modules in standard library.
  import os
  import sys

  # Modules from third-party.
  import numpy as np

  # Modules in the current project.
  import modmesh as mm
  from modmesh import onedim

  # Explicit relative import is OK.
  from . import core

.. note::

  ``modmesh`` can be shorthanded as ``mm``.

Do not import multiple modules in one line:

.. code-block:: python

  # BAD BAD BAD
  import os, sys

Never do implicit relative import:

.. code-block:: python

  # BAD for modules in the current project.
  import onedim

Integer Type
====

Use fixed-width integers (``int32_t``, ``uint8_t``, etc.) Do not use the basic
integer types (``int``, ``long``, etc.) unless there is not another choice.

C++ Comment
====

Comment blocks follow `the doxygen style guidelines
<https://www.doxygen.nl/manual/docblocks.html>`__ if convenient.

If possible, provide references to literature or documents in comments.

C++ Include File
====

The inclusion guard uses ``#pragma once`` in the first line before everything.

Always use path-first inclusion (angle branket). Do not use current-first
(double quote).

.. code-block:: cpp

  // Use this: search for include file start with the paths to the compiler.
  #include <modmesh/base.hpp>
  // Do not use this. This starts to search from the directory of the file.
  #include "modmesh/toggle.hpp"

C++ Namespace
====

Put everything in the ``modmesh`` namespace.

Never ``using namespace`` outside a local scope (like a function). Another
namepsace is not a local scope and should not ``using namespace``. When
accessing something in a namespace (e.g., ``modmesh``) from outside, spell
out the full name:

.. code-block:: cpp

  // An anonymouse namespace
  namespace
  {

  modmesh::real_type local_function(modmesh::int_type value);

  } /* end namespace */

The namespace ``modmesh`` may be aliased to ``mm`` in a local scope. No alias
should be use outside a local scope.

.. code-block:: cpp

  modmesh::real_type local_function(modmesh::int_type value)
  {
      // Alias the modmesh namespace to mm.
      namespace mm = modmesh;
      return mm::real_type(value); // Same as modmesh::real_type(value);
  }

Needless to say that ``using namespace std;`` is absolutely forbidden.

Implementation Detail
----

Name the namespace for implementation details to ``detail``.

.. code-block:: cpp

  namespace modmesh
  {

  namespace detail
  {
      // Implementation detail
  } /* end namespace detail */

  } /* end namespace modmesh */

C Pre-Processor Macro
====

Prefix macros with ``MM_DECL_``. If they are not supposed to be used as a
global helper, delete them after consumption.

C++ Standard
====

Use C++-17 and beyond.

Follow the `rule of five
<https://en.cppreference.com/w/cpp/language/rule_of_three>`__. Most of the time
just spell out all default implementation of constructors and assignment
operators and group them together:

.. code-block:: cpp

  class MyClass
  {
  public:
      // Listing all default implementation will make the intention clear and
      // it is easier to change from default to delete.

      // Default constructor.
      MyClass() = default;
      // Copy constructor.
      MyClass(MyClass const &) = default;
      // Move constructor.
      MyClass(MyClass &&) = default;
      // Copy assignment operator.
      MyClass & operator=(MyClass const &) = default;
      // Move assignment operator.
      MyClass & operator=(MyClass &&) = default;
      // Destructor.
      ~MyClass() = default;
  }; /* end class MyClass */

C++ Encapsulation
====

Prefer encapsulated classes over POD struct so that we always provide
accessors. We provide accessors for even scalars of fundamental types.

.. code-block:: cpp

  class MyPowerHouse
  {

  public:

      void calculate_internal_data();

      // Use the same-name style for accessors.
      double internal_value() const { return m_internal_value; }
      double & internal_value() { return m_internal_value; }

      // It may be good to have a blank line between accessor pairs.
      SimpleArray<double> const & internal_data() const { return m_internal_data; }
      SimpleArray<double> & internal_data() { return m_internal_data; }

  private:

      double m_internal_value = 0.0;
      SimpleArray<double> m_internal_data;

  }; /* end class MyPowerHouse */

Prefer Same-Name Accessors
----

(Python does not need accessors. Do not add accessors in Python code.)

Prefer same-name accessors because we expose a lot of internal containers:

.. code-block:: cpp

  // Getter is const and return a copy of a fundamental type.
  double internal_value() const { return m_internal_value; }
  // Setter is non-const and return a reference.
  double & internal_value() { return m_internal_value; }

  // Getter is const and return a const reference of a non-fundamental type.
  SimpleArray<double> const & internal_data() const { return m_internal_data; }
  // Setter is non-const and return a reference.
  SimpleArray<double> & internal_data() { return m_internal_data; }

Sometimes we may use the getter-and-setter style to supplement the same-name
accessors:

.. code-block:: cpp

  // Getter is const and return a copy of a fundamental type.
  double get_internal_value() const { return m_internal_value; }
  // Setter takes
  void set_internal_value(double v) { m_internal_value = v; }

  // Getter is const and return a const reference of a non-fundamental type.
  SimpleArray<double> const & internal_data() const { return m_internal_data; }
  // Setter is non-const and return a reference.
  SimpleArray<double> & internal_data() { return m_internal_data; }

It is OK for accessors of the same-name and getter-and-setter styles to be
available for the same member datum, but we should only do it when necessary.

C++ Ending Mark
====

Add ending marks to classes and namespaces.  They are usually too long (across
hundreds of lines) to keep track of.

.. code-block:: cpp

  namespace modmesh
  {

  class MyClass
  {
      // Code.
  }; /* end class MyClass */

  } /* end namespace modmesh */

Copyright Notice
====

modmesh uses the `BSD license <http://opensource.org/licenses/BSD-3-Clause>`__.
When creating a new file, put the following text at the top of the file
(replace ``<Year>`` with the year you create the file and ``<Your Name>`` with
your name and maybe email).  The license text formatted for C++ files:

.. code-block:: cpp

  /*
   * Copyright (c) <Year>, <Your Name>
   *
   * Redistribution and use in source and binary forms, with or without
   * modification, are permitted provided that the following conditions are met:
   *
   * - Redistributions of source code must retain the above copyright notice,
   *   this list of conditions and the following disclaimer.
   * - Redistributions in binary form must reproduce the above copyright notice,
   *   this list of conditions and the following disclaimer in the documentation
   *   and/or other materials provided with the distribution.
   * - Neither the name of the copyright holder nor the names of its contributors
   *   may be used to endorse or promote products derived from this software
   *   without specific prior written permission.
   *
   * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   * POSSIBILITY OF SUCH DAMAGE.
   */

The license text formatted for Python files:

.. code-block:: python

  # -*- coding: UTF-8 -*-
  #
  # Copyright (c) <Year>, <Your Name>
  #
  # All rights reserved.
  #
  # Redistribution and use in source and binary forms, with or without
  # modification, are permitted provided that the following conditions are met:
  #
  # - Redistributions of source code must retain the above copyright notice, this
  #   list of conditions and the following disclaimer.
  # - Redistributions in binary form must reproduce the above copyright notice,
  #   this list of conditions and the following disclaimer in the documentation
  #   and/or other materials provided with the distribution.
  # - Neither the name of the copyright holder nor the names of its contributors
  #   may be used to endorse or promote products derived from this software
  #   without specific prior written permission.
  #
  # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
  # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  # POSSIBILITY OF SUCH DAMAGE.

.. vim: set ft=rst ff=unix fenc=utf8 et sw=2 ts=2 sts=2:
