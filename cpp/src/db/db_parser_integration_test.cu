/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <db/db_parser_integration_test.cuh>
#include <stdio.h>
#include <iostream>

namespace cugraph {
namespace db {
std::string getParserVersion() {
  std::string version = libcypher_parser_version();
  return version;
}

void printOutAst(std::string input) {
  const cypher_parse_result_t* result = cypher_parse(input.c_str(), NULL, NULL, 0);
  cypher_parse_result_fprint_ast(result, stdout, 80, NULL, 0);

  uint32_t numErrors = cypher_parse_result_nerrors(result);
  if (numErrors > 0) {
    std::cout << "There are " << numErrors << " parse errors in the query string.\n";
    for (uint32_t i = 0; i < numErrors; i++){
      const cypher_parse_error_t* error = cypher_parse_result_get_error(result, i);
      const char* errorMsg = cypher_parse_error_message(error);
      std::cout << "Error " << i << " Message: " << errorMsg << "\n";
    }
  }
  else {
    uint32_t numRoots = cypher_parse_result_nroots(result);
    std::cout << "\n\nThere are " << numRoots << " AST roots in the result.\n";
    for (uint32_t i = 0; i < numRoots; i++) {
      const cypher_astnode_t* root = cypher_parse_result_get_root(result, i);
      cypher_astnode_type_t type = cypher_astnode_type(root);
      const char* type_desc = cypher_astnode_typestr(type);
      uint32_t n_child = cypher_astnode_nchildren(root);
      std::cout << "Root " << i << " which is a: " << type_desc << " and has " << n_child
          << " children\n";
      const cypher_astnode_t* child = cypher_astnode_get_child(root, 0);
      const cypher_astnode_t* body = cypher_ast_statement_get_body(root);
      if (child == body)
        std::cout << "Body is equal to child for statement.\n";
      else
        std::cout << "Body is not equal to child for statement.\n";
      cypher_astnode_type_t cType = cypher_astnode_type(child);
      const char* ctype_desc = cypher_astnode_typestr(cType);
      std::cout << "Statement child is a " << ctype_desc << " node\n";
    }
  }
}

} // namespace db
} //namespace cugraph
