#!/usr/bin/python

import os, sys, re, glob

from os.path import join, split, splitext

def process_file(project_base_dir, fobj, filename, base_list, project_name) :
    enum_ids = [];
    policy_tag_str = 'template\s*?<\s*?typename DerivedPolicy.*?>.*?;';
    inc_filename = filename.replace(project_base_dir + "/", "");

    with open(filename, "r") as fobj0:
        text = fobj0.read();
        output = re.findall(policy_tag_str, text, re.DOTALL);

        if output :

            base_dir,base_file = split(filename);
            start_dir = os.path.split(base_dir)[1];

            if base_file == "copy.h" and project_name == "cusp" :
                return;

            impl_file_list = [join(base_dir, "detail", splitext(base_file)[0] + ".inl")];

            if base_file == "copy.h" and project_name == "thrust" :
                impl_file_list.extend([join(base_dir, "detail", splitext(base_file)[0] + "_if.inl")]);

            for impl_file in impl_file_list:
                print "processing {}".format(impl_file);
                if os.path.isfile(impl_file):
                    with open(impl_file, "r") as fobj1:

                        ext_policy_tag_str = 'template\s*?<\s*?typename DerivedPolicy.*?>.*?\{.*?\}';
                        text = fobj1.read();
                        output = re.findall(ext_policy_tag_str, text, re.DOTALL);

                        if output :
                            fobj.write("#include <{}>\n".format(inc_filename));

                            for routine in output :
                                if '::execution_policy<' in routine :
                                    continue;

                                name_search = re.search("using.*::(?P<name>\w+);", routine);
                                if name_search :
                                    name = name_search.group("name")
                                else :
                                    continue;

                                routine = re.sub("typename\s*?DerivedPolicy,\s*?", "", routine);
                                routine = re.sub("__host__\s*__device__", "", routine);
                                routine = re.sub("const thrust::detail::execution_policy_base<DerivedPolicy>", "my_policy", routine);
                                routine = re.sub("thrust::detail::derived_cast\(thrust::detail::strip_const\(exec\)\)",
                                                 "exec.get()" if name not in base_list else "exec.base()", routine);

                                ret = re.search(r"template\s*?<.*?>\s*(?P<return_type>[\w\d,:<> ]+)\s*" + name + r"\s*?\(", routine, re.DOTALL);
                                if not ret:
                                    raise ValueError("\n"+routine)
                                return_type = ret.group("return_type").strip();

                                if "blas" in routine :
                                    name = "_".join(["blas", name]);

                                lines = routine.split("\n")

                                out_lines = []
                                for line in lines:
                                    if line :
                                        if "return" in line :
                                            enum_id = "__{}_{}__".format(project_name.upper(), name.upper());
                                            if enum_id not in enum_ids :
                                                enum_ids.append(enum_id);
                                            out_lines.append("\n  exec.start({});".format(enum_id));
                                            if "void" in return_type:
                                                out_lines.append("  " + line.split("return ",1)[1]);
                                            else :
                                                out_lines.append("  {} ret = {}".format(return_type, line.split("return ",1)[1]));
                                            out_lines.append("  exec.stop();");

                                            if "void" not in return_type:
                                                out_lines.append("\n  return ret;");
                                        else :
                                            out_lines.append(line)


                                routine = "\n".join(out_lines);
                                fobj.write("\n" + routine + "\n\n");

    return enum_ids;

def generate(base_dir, project_name, dir_list, base_list, start_index) :
    # find all .inls base directories
    sources = [];
    directories = [os.path.join(base_dir, dir_name) for dir_name in dir_list];
    extensions = ['*.h'];

    for dir in directories:
      for ext in extensions:
        regexp = os.path.join(dir, ext)
        sources.extend(glob.glob(regexp))

    local_ids = []
    with open("my_{}_func.h".format(project_name), "w") as fobj:
        for source in sources :
            ids = process_file(base_dir, fobj, source, base_list, project_name);
            if ids :
                local_ids.extend(ids);

    return local_ids

if __name__ == "__main__" :

    thrust_base = os.path.join(os.environ["HOME"], "thrust");
    thrust_base_funcs = ["for_each", "for_each_n", "inclusive_scan", "exclusive_scan", "reduce"];
    global_ids = generate(thrust_base, "thrust", ["thrust"], thrust_base_funcs, 0);

    cusp_base = os.path.join(os.environ["HOME"], "cusplibrary");
    cusp_base_funcs = [];
    ids = generate(cusp_base, "cusp", ["cusp", "cusp/krylov", "cusp/graph", "cusp/lapack"], cusp_base_funcs, 0);
    global_ids.extend(ids);

    with open("my_policy_map.h", "w") as fobj:
        fobj.write("enum {\n");
        fobj.write(",\n".join(global_ids));
        fobj.write("\n};\n\n");

        fobj.write("static const char * ARR_NAMES[] = {\n");
        fobj.write(",\n".join(["\"" + k.lower().strip('__') + "\"" for k in global_ids]));
        fobj.write("\n};\n\n");

