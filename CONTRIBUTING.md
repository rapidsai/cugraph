# Contributing to cuGraph
cuGraph, and all of RAPIDS in general, is an open-source project where we encourage community involvement.  There are multiple ways to be involved and contribute to the cuGraph community, the top three paths are listed below:

#### 1) Filing in Issue for the RAPIDS cuGraph team to work

***Bug Report***
If you notice something not working please file an issue

-	File an [issue](https://github.com/rapidsai/cugraph/issues/new/choose)
-	Select **Bug** Report
-	describing what you encountered and the severity of the issue:  Does code crash or just not return the correct results
-	Include a sequence of step to reproduce the error
-	The RAPIDS team will evaluate and triage the issue
-	If you believe the issue needs priority attention, please include that in 
    the issue to notify the team.

***Propose a new Feature or Enhancement***
If there is a feature or enhancement to an existing feature, please file an issue

-	File an [issue](https://github.com/rapidsai/cugraph/issues/new/choose)
-	Select either **Enhancement Request** or **Feature Report**
-	describing what you want to see added or changed.  For new features, if there is a white paper on the analytic, please include a reference to it
-	The RAPIDS team will evaluate and triage the issue, and then schedule it in a future release.
-	If you believe the issue needs priority attention, please include that in 
    the issue to notify the team.

***Ask a Question***
There are several ways to ask questions, including [Stack Overflow]( https://stackoverflow.com/)  or the RAPIDS [Google forum]( https://groups.google.com/forum/#!forum/rapidsai), but an GitHub issue can be filled.  

-	File an [issue](https://github.com/rapidsai/cugraph/issues/new/choose)
-	Select Question
-	describing your question
-	The RAPIDS team will attempt to answer your question as quick as possible


#### 2) Propose a New Feature and Implement It

We love when people want to get involved, and if you have a suggestion for a new feature or enhancement and want to be the one doing the development work, we fully encourage that.  

-  Submit a New Feature Issue (see above) and state that you are working on it.
- The team will give feedback on the issue and happy to make suggestions
- Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.


#### 3) You want to implement a feature or bug-fix for an outstanding issue
- Follow the [code contributions](#code-contributions) guide below.

If you need more context on a particular issue, please ask.


# So You Want to Contribute Code

### TL;DR General Development Process

1. Read the documentation on [building from source](SOURCEBUILD.md) to learn how to setup, and validate, the development environment
2. Find an issue, or submit an issue, to work on. 
3. Comment on the issue saying you are going to work on it
4. Fork the cuGraph [repo](#fork) and Code (make sure to add unit tests)!
5. When done, and code passes local CI, create your pull request (PR)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a RAPIDS developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!



### Fork a private copy of cuGraph that can we modified
<a name="fork"></a>

The RAPIDS cuGraph repo cannot directly be modified.  Contributions must come in the form of a *Pull Request* from a folked version of cugraph.    GitHub as a nice write up ion the process:  https://help.github.com/en/github/getting-started-with-github/fork-a-repo

Read the section on [building cuGraph from source](SOURCEBUILD.md) to validate that the environment is correct.  


### Development Environment

There is no recommended or preferred development environment.  There are a few *must have* conditions on GPU hardware and library versions.  But for the most part, users can work in the environment that they are familiar and comfortable with.  

**Hardware**

* You need to have accesses to an NVIDAI GPU that is Pascal or later.


**IDEs**

There is no recommended IDE, here is just a list of what cuGraph developers currently use (not in any priority order)

* NSIGHT
* Eclipse (with the C++ and Python modules)
* VSCode
* VIM / VI (old school programming)
  * With plug-ins like [FZF](https://github.com/junegunn/fzf), [Rg](https://github.com/BurntSushi/ripgrep)


Using VSCode, you can develop remotely from the hardware if you so wish.  Alex Fender has a setting up remote development:  https://github.com/afender/cugraph-vscode


**Debug**

* cuda-memcheck


A debug launch can also be enabled in VSCode with something like:  https://github.com/harrism/cudf-vscode/blob/master/.vscode/launch.json


### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/cugraph/projects).

> **Pro Tip:** Always look at the release board with the lowest number for
issues to work on. This is where RAPIDS developers also focus their efforts.  cuGraph maintains a project board for the current release plus out two future releases.  This allows to better long term planning

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.







