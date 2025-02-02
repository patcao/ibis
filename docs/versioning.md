# Versioning Policy

Ibis follows a [Semantic Versioning](https://semver.org/) scheme
(`MAJOR.MINOR.PATCH`, like `6.1.0`).

- An increase in the `MAJOR` version number will happen when a release contains
  breaking changes in the public API. This includes anything documented in the
  [reference documentation](./reference/expressions/index.md), excluding any
  features explicitly marked as "experimental". Features not part of the public
  API (e.g. anything in `ibis.expr.operations` may make breaking changes at any
  time).

- An increase in the `MINOR` or `PATCH` version number indicate changes to
  public APIs that should remain compatible with previous Ibis versions with
  the same `MAJOR` version number.

## Supported Python Versions

Ibis follows [NEP29](https://numpy.org/neps/nep-0029-deprecation_policy.html)
with respect to supported Python versions.

This has been in-place [since Ibis version 3.0.0](https://github.com/ibis-project/ibis/blob/5015677d78909473014a61725d371b4bf772cdff/docs/blog/Ibis-version-3.0.0-release.md?plain=1#L83).

The [support
table](https://numpy.org/neps/nep-0029-deprecation_policy.html#support-table)
shows the schedule for dropping support for Python versions.

The next major release of Ibis that occurs on or after the NEP29 drop date
removes support for the specified Python version.
