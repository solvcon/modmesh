# Coding Style

## Naming

1. Class names use CamelCase, for example: `class CallProfiler;`.
   - Qt-related classes prepend an additional `R`, like `class RLine;`.
     * A class is considered Qt-related if its parent class belongs to the Qt library.

2. Variable names and using-declarations use snake_case:
   - Example variable: `R3DWidget * viewer;`
   - Example using-declaration: `using size_type = std::size_t;`

3. Qt-related function names use CamelCase, while non-Qt-related function names use snake_case.
   - Functions are Qt-related if they belong to a Qt-related class.
   - Example Qt-related function: `QMdiSubWindow * RManager::addSubWindow(Args &&... args);`
   - Example non-Qt-related function: `void initialize_buffer(pybind11::module & mod);`

4. Class members begin with `m_`.

5. [Macros](https://en.cppreference.com/w/cpp/preprocessor/replace) start with `MM_DECL_`.

6. [Using-declarations](https://en.cppreference.com/w/cpp/language/using_declaration) end with `_type`.

7. Acronyms within names should be treated as words instead of using ALL_CAPS_CONSTANTS.
   - Example classes: `class HttpRequest;`
   - Example function: `void update_geometry_impl(StaticMesh const & mh, Qt3DCore::QGeometry * geom);`

Certainly! Here's a more coherent version:

## Comments

1. Comment blocks should follow [Doxygen style guidelines](https://www.doxygen.nl/manual/docblocks.html).

2. For one-line comments like `/* end of NamespaceOrClassName */`, 
   place them at the end of namespaces or classes.  
   Example usage:
   ```c++
   namespace modmesh
   {
   // More code...
   }; /* end of modmesh */
   ```

3. It is highly recommended to include references in comment blocks.

## Formatting

1. Ensure there is a blank line between the definitions of classes and functions.

2. Long lines should be divided into shorter ones.
   - For C++, no specific line length limit is set.
   - For Python, adhere to an 80-character limit per line.

## Class

1. It's advisable to explicitly define constructors and destructors for classes whenever possible.