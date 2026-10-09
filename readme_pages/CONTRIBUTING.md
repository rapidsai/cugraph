# Contributing to cuGraph
cuGraph is an open-source project where we encourage community involvement.

There are multiple ways to be involved and contribute to the cuGraph community, the top paths are listed below:

* [File a New Issue](#new_issue)
* [Propose a New Feature](#new_feature)
* [Work on an Existing Issue](#existing_issue)

If you are ready to contribute, jump right to the [Contribute Code](#so-you-want-to-contribute-code) section.


__Style Formatting Tools:__
* `clang-format`  version 16.0+
* `flake8`        version 6.0.0+




## 1) File a New Issue<a name="new_issue"></a>
To file an issue, go to the RAPIDS cuGraph [issue](https://github.com/rapidsai/cugraph/issues/new/choose) page and select the appropriate issue type.  Once an issue is filed the RAPIDS cuGraph team will evaluate and triage the issue.  If you believe the issue needs priority attention, please include that in the issue to notify the team.


***Bug Report***: If you notice something not working please file an issue
-	Select **Bug** Report
-	Describing what you encountered and the severity of the issue:  Does code crash or just not return the correct results
-	Include a sequence of step to reproduce the error

***Propose a new Feature or Enhancement***: If there is a feature or enhancement to an existing feature, please file an issue

-	Select either **Enhancement Request** or **Feature Report**
-	describing what you want to see added or changed.  For new features, if there is a white paper on the analytic, please include a reference to it

***Ask a Question***: If you need help or have a question

-	Select Question
-	describing your question

Additionally, you can ask questions on [Stack Overflow](https://stackoverflow.com/).

## 2) Propose a New Feature and Implement It <a name="new_feature"></a>

We love when people want to get involved, and if you have a suggestion for a new feature or enhancement and want to be the one doing the development work, we fully encourage that.

- Submit a New Feature Issue (see above) and state that you are working on it.
- The team will give feedback on the issue and happy to make suggestions
- Once we agree that the plan looks good, go ahead and implement it
- Follow the [code contributions](#so-you-want-to-contribute-code) guide below.


## 3) Work on an Existing Issue <a name="existing_issue"></a>
- Find an open Issue, and post that you would like to work that issues
- Once we agree that the plan looks good, go ahead and implement it
- Follow the [code contributions](#so-you-want-to-contribute-code) guide below.

If you need more context on a particular issue, please ask.

<br>

----


# So you want to contribute code<a name="so-you-want-to-contribute-code"></a>

**TL;DR General Development Process**
1. Read the documentation on [building from source](../docs/cugraph/source/installation/source_build.md) to learn how to setup, and validate, the development environment
2. Read the RAPIDS [Code of Conduct](https://docs.nvidia.com/datascience/resources/conduct/)
3. Find or submit an issue to work on (include a comment that you are working issue)
4. Fork the cuGraph [repo](#fork) and Code (make sure to add unit tests)!
5. All RAPIDS projects are released under the Apache-2.0 license, so also make sure all source files that support comments include a copyright and the Apache-2.0 license text.
6. When done, and code passes local CI, create your pull request (PR)
   1. Ensure the code matches our style by running `pre-commit` on your changed files
7. Verify that cuGraph CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
8. Wait for other developers to review your code and update code as needed
9. Once reviewed and approved, a RAPIDS developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

**The _FIXME_ comment**

Use the _FIXME_ comment to capture technical debt.  It should not be used to flag bugs since those need to be cleaned up before code is submitted.
Usage of TODO or any other tag will not be accepted.



## Fork a private copy of cuGraph <a name="fork"></a>
The RAPIDS cuGraph repo cannot directly be modified.  Contributions must come in the form of a *Pull Request* from a forked version of cugraph.  GitHub has a nice write-up on the process: https://help.github.com/en/github/getting-started-with-github/fork-a-repo

1. Fork the cugraph repo to your GitHub account
2. clone your version
```git clone https://github.com/<YOUR GITHUB NAME>/cugraph.git```


Read the section on [building cuGraph from source](../docs/cugraph/source/installation/source_build.md) to validate that the environment is correct.

**Pro Tip** add an upstream remote repository so that you can keep your forked repo in sync
```git remote add upstream https://github.com/rapidsai/cugraph.git```

3. Checkout the latest branch
cuGraph only allows contribution to the current default branch.
Please check the [cuGraph](https://github.com/rapidsai/cugraph) page for the name of the current branch.

```shell
git checkout main
```

4. Make your code changes
5. Once your code works and passes tests, commit and push your changes

   ```shell
   git add <changed files>
   git commit
   git push
   ```

6. From the GitHub web page, open a Pull Request

### Development Environment

There is no recommended or preferred development environment.  There are a few *must have* conditions on GPU hardware and library versions.  But for the most part, users can work in the environment that they are familiar and comfortable with.

**Hardware**

* You need to have access to an NVIDIA GPU that is Volta or later.


**IDEs**

There is no recommended IDE, here is just a list of what cuGraph developers currently use (not in any priority order)

* NSIGHT
* Eclipse (with the C++ and Python modules)
* VSCode
* VIM / VI (old school programming)
  * With plug-ins like [FZF](https://github.com/junegunn/fzf), [Rg](https://github.com/BurntSushi/ripgrep)


**Debug**

* cuda-memcheck
* cuda-gdb


A debug launch can also be enabled in VSCode with something like:  https://github.com/harrism/cudf-vscode/blob/master/.vscode/launch.json


### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/cugraph/projects).

> **Pro Tip:** Always look at the release board with the lowest number for
issues to work on. This is where RAPIDS developers also focus their efforts.  cuGraph maintains a project board for the current release plus its two future releases.  This allows for better long-term planning.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.


### Style Guide
All Python code most pass flake8 style checking
All C++ code must pass clang style checking

### Tests
All code must have associate test cases.  Code without test will not be accepted
