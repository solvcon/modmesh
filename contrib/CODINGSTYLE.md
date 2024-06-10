# Coding Style
## Introduction
This is the coding style guide for Modmesh.  
* [Introduction](#introduction)
* [Naming](#naming)
* [Comments](#comments)
* [Formatting](#formatting)
* [Class](#class)
* [Functions](#functions)

### Naming
1. The class names use Camel case, such as ```class R3DWidget;```. 
2. The name of the variable / using-declarations uses snake case,  
such as ```R3DWidget * viewer;``` and ```using size_type = std::size_t;```. 
3. The QT-related function name uses Camel case,  
whereas non-QT-related function name uses snake case.
4. The name of the class member starts with ```m_```.
5. The name of [macros](https://en.cppreference.com/w/cpp/preprocessor/replace) starts with ```DECL_MM_```.
6. The [using-declarations](https://en.cppreference.com/w/cpp/language/using_declaration) ends with ```_type```.

### Comments
1. The comment blocks follow the [doxygen styles](https://www.doxygen.nl/manual/docblocks.html).  
2. The one-line comment, ```/* end of NamespaceOrClassName */```,  
should be appended to the end of a namespace or class.  
For example :
```c++
namespace modmesh
{
// More code...
}; /* end of modmesh */
```
3. It is highly recommended to add the reference in the comment block.

### Formatting
1. It is necessary to obey the [clang-format](https://clang.llvm.org/docs/index.html), or you must fail the github action.  
2. There exists a space between the definition of the classes/functions.
3. A long line should be devided to several shorter lines.

### Class
1. The class should follow [the rule of five](https://en.cppreference.com/w/cpp/language/rule_of_three).

### Functions
1. The recursive function should be replaced with the iterative function, unless it is inevitable to be used.