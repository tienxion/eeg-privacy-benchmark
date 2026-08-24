# Security Policy

## Supported version

Security fixes are accepted for the current public v1 release line.

## Reporting a vulnerability

Before a public remote exists, report vulnerabilities privately to the project
owner through an agreed private channel. After publication, use the repository
host's private security-advisory mechanism when available. Do not include raw
EEG, participant data, credentials, or exploit details in a public issue.

Scientific evidence that a measured privacy attack succeeds is normally a
benchmark result, not a software vulnerability. Implementation flaws that leak
files, execute untrusted input, expose credentials, or invalidate stated
boundaries should be handled as security reports.

The secure-aggregation module is a functional simulator, not reviewed
cryptographic software. It must not be used as a production security control.
