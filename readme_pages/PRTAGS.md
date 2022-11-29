# Pull Request Tags
If you look at the list of current [Pull Request](https://github.com/rapidsai/cugraph/pulls) you will notice a set of bracketed tags in the subject line. Those tags help developers focus attention and know what is being asked.  

PR = Pull Request

|  TAG       |                                                       |
|------------|-------------------------------------------------------|
| WIP        | _Work In Progress_ - While it would be perferred to simple place the PR is [DRAFT](https://github.blog/2019-02-14-introducing-draft-pull-requests/) state (through GitHub), you can also label the PR as being a work in progress.  Within the RAPIDS cuGraph team, we try to open a PR when development starts.  This allows other to review code as it is being developed and provide feedback before too much code needs to be refactored.  It also allows process to be tracked.  __A WIP PR will not be merged into baseline__ |
| skip-ci    | _Do Not Run CI_ - This flag prevents CI from being run.  It is good practice to include this with the **WIP** tag since code is typically not at a point where it will pass CI.  |
| skip ci    | same as above                                          |
| REVIEW     | The code is ready for a full code review.  Only code that has passed a code review is merged into the baseline  |


Code must pass CI before it is merged