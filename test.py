import json

# Define your JSON data as a Python dictionary
data = {
    "Network_Security": {
        "domain_name": "Network Security",
        "keywords": [
            "Firewall",
            "Intrusion Detection System (IDS)",
            "Intrusion Prevention System (IPS)",
            "VPN",
            "Network Segmentation",
            "Encryption",
            "Access Control",
            "Traffic Monitoring",
            "Malware Protection",
            "Threat Intelligence"
        ],
        "preventive_measures": [
            "Implement firewalls to control incoming and outgoing traffic.",
            "Use intrusion detection and prevention systems (IDPS) to monitor network activity.",
            "Establish Virtual Private Networks (VPNs) for secure remote access.",
            "Segment the network to limit access between different areas."
        ],
        "followup_questions": [
            "What measures are currently in place to prevent lateral movement within the network? Are there any specific segments that are particularly vulnerable?",
            "Can you provide details on the current firewall models in use? When was the last time they were updated or replaced, and what are the barriers to upgrading them?",
            "How are unauthorized access attempts currently being monitored and reported? What actions are taken when such attempts are detected?",
            "What kind of training programs are in place for employees regarding network security best practices? Are there plans to implement additional training sessions to address current gaps?"
        ],
        "risk_indicators": [
            "Increased number of unauthorized access attempts.",
            "Outdated or unpatched firewall systems.",
            "Lack of network segmentation leading to potential lateral movement.",
            "Insufficient employee training on security protocols."
        ],
        "calculated_risk_score": 75.0,
        "compliance_requirements": {
            "GDPR": [
                "Data encryption for network traffic",
                "Access control logs"
            ],
            "NIST": [
                "Network monitoring tools",
                "Incident response plan"
            ],
            "ISO_27001": [
                "Firewall configuration",
                "Regular security audits"
            ]
        }
    },
    "Endpoint_Security": {
        "domain_name": "Endpoint Security",
        "keywords": [
            "Antivirus Software",
            "Endpoint Detection and Response (EDR)",
            "Patch Management",
            "Device Control",
            "Application Whitelisting",
            "Mobile Device Management (MDM)",
            "Threat Hunting",
            "Behavioral Analysis",
            "Data Encryption",
            "User Education"
        ],
        "preventive_measures": [
            "Deploy endpoint detection and response (EDR) tools across all devices.",
            "Ensure all endpoints have up-to-date antivirus software.",
            "Implement mobile device management (MDM) solutions to secure mobile devices.",
            "Regularly update and patch all operating systems and applications."
        ],
        "followup_questions": [
            "How often are endpoints scanned for vulnerabilities, and what tools are used for this purpose?",
            "What is the process for responding to detected threats on endpoints?",
            "Are there any exceptions in place for device security measures, and how are these managed?",
            "What user awareness training is provided to ensure secure usage of endpoints?"
        ],
        "risk_indicators": [
            "High number of malware infections detected on endpoints.",
            "Unmanaged or outdated devices accessing the network.",
            "Inconsistent patch management practices.",
            "Low user awareness of endpoint security protocols."
        ],
        "calculated_risk_score": 58.33,
        "compliance_requirements": {
            "NIST SP 800-171": [
                "Regular patching and vulnerability scanning",
                "Endpoint detection and response (EDR) deployment"
            ],
            "ISO_27001": [
                "Endpoint security policies",
                "User access management"
            ],
            "PCI DSS": [
                "Antivirus installation and management",
                "Secure device configuration"
            ]
        }
    },
    "Cloud_Security": {
        "domain_name": "Cloud Security",
        "keywords": [
            "Data Encryption",
            "Access Control Policies",
            "Cloud Security Posture Management (CSPM)",
            "Identity and Access Management (IAM)",
            "Secure APIs",
            "Data Loss Prevention (DLP)",
            "Compliance Standards (e.g., GDPR, HIPAA)",
            "Virtual Private Cloud (VPC)",
            "Security Audits",
            "Shared Responsibility Model"
        ],
        "preventive_measures": [
            "Use data encryption for data at rest and in transit.",
            "Implement strict identity and access management (IAM) policies for cloud services.",
            "Regularly audit cloud configurations to prevent misconfigurations.",
            "Secure APIs used for cloud services against unauthorized access."
        ],
        "followup_questions": [
            "What specific encryption methods are employed for sensitive data in the cloud?",
            "How often are cloud services and configurations audited for compliance and security?",
            "Can you describe your incident response plan for cloud security breaches?",
            "What processes are in place to manage and monitor third-party access to cloud resources?"
        ],
        "risk_indicators": [
            "Frequent cloud configuration errors.",
            "Unauthorized access attempts to cloud services.",
            "Lack of encryption for sensitive data stored in the cloud.",
            "Insufficient monitoring of cloud usage and access."
        ],
        "calculated_risk_score": 58.33,
        "compliance_requirements": {
            "ISO_27017": [
                "Cloud-specific security controls",
                "Data encryption for cloud storage"
            ],
            "NIST SP 800-144": [
                "Cloud configuration audits",
                "Access control policies"
            ],
            "SOC 2": [
                "Third-party risk assessments",
                "Data protection and privacy controls"
            ]
        }
    },
    "Application_Security": {
        "domain_name": "Application Security",
        "keywords": [
            "Common Vulnerabilities and Exposures (CVE)",
            "Common Vulnerability Scoring System (CVSS)",
            "Security Information and Event Management (SIEM)",
            "Risk Assessment",
            "Security Testing",
            "Code Review",
            "Secure Coding Practices",
            "Web Application Firewall (WAF)",
            "Penetration Testing",
            "Input Validation"
        ],
        "preventive_measures": [
            "Conduct regular vulnerability assessments and penetration testing.",
            "Implement application firewalls to filter and monitor HTTP traffic.",
            "Follow secure coding practices during application development.",
            "Ensure timely application of security patches and updates."
        ],
        "followup_questions": [
            "What is your process for identifying and remediating application vulnerabilities?",
            "How often do you conduct penetration tests, and what were the findings of the last test?",
            "Are developers trained in secure coding practices, and how is this enforced?",
            "What is the current status of application patch management across your systems?"
        ],
        "risk_indicators": [
            "Increased number of detected vulnerabilities in applications.",
            "Delayed application patching resulting in security gaps.",
            "Poor adherence to secure coding standards by developers.",
            "Lack of testing for application security before deployment."
        ],
        "calculated_risk_score": 56.67,
        "compliance_requirements": {
            "OWASP": [
                "Secure coding guidelines",
                "Regular vulnerability assessments"
            ],
            "ISO_27034": [
                "Application security management",
                "Risk assessment procedures"
            ],
            "NIST": [
                "Application vulnerability management",
                "Patch management practices"
            ]
        }
    },
    "Identity_and_Access_Management_IAM": {
        "domain_name": "Identity and Access Management (IAM)",
        "keywords": [
            "Multi-Factor Authentication (MFA)",
            "Single Sign-On (SSO)",
            "User Provisioning",
            "Role-Based Access Control (RBAC)",
            "Access Rights",
            "Identity Governance",
            "Privileged Access Management (PAM)",
            "Authentication",
            "Directory Services",
            "Audit Trails"
        ],
        "preventive_measures": [
            "Implement multi-factor authentication (MFA) across all critical systems.",
            "Regularly review and update access control policies.",
            "Utilize identity federation for secure single sign-on (SSO).",
            "Manage privileged access using a separate process."
        ],
        "followup_questions": [
            "What challenges have you faced in implementing MFA for all users?",
            "How often do you review user access rights and roles?",
            "Can you describe your process for provisioning and de-provisioning user accounts?",
            "What measures are in place to monitor and manage privileged accounts?"
        ],
        "risk_indicators": [
            "High number of unauthorized access attempts.",
            "Ineffective management of user access rights.",
            "Delayed de-provisioning of terminated employees' accounts.",
            "Weak implementation of multi-factor authentication."
        ],
        "calculated_risk_score": 71.67,
        "compliance_requirements": {
            "NIST": [
                "Multi-factor authentication implementation",
                "Access control audits"
            ],
            "ISO_27001": [
                "User provisioning and de-provisioning controls",
                "Privileged access management"
            ],
            "SOC 2": [
                "Audit trails for access management",
                "Regular review of access rights"
            ]
        }
    },
    "Data_Security": {
        "domain_name": "Data Security",
        "keywords": [
            "Data Classification",
            "Encryption at Rest and in Transit",
            "Data Masking",
            "Access Controls",
            "Backup and Recovery Plans",
            "Data Loss Prevention (DLP)",
            "Secure Data Disposal",
            "Regulatory Compliance (e.g., GDPR, CCPA)"
        ],
        "preventive_measures": [
            "Implement encryption for sensitive data both at rest and in transit.",
            "Establish data loss prevention (DLP) strategies to monitor and protect data.",
            "Regularly classify data to determine appropriate protection measures.",
            "Conduct routine backups of critical data to ensure recoverability."
        ],
        "followup_questions": [
            "What data classification processes do you currently have in place?",
            "How often are backups performed, and where are they stored?",
            "What encryption standards are used for sensitive data?",
            "Are there any recent incidents of data breaches, and what was learned from them?"
        ],
        "risk_indicators": [
            "Unencrypted sensitive data found in storage.",
            "Frequent incidents of data leakage.",
            "Inadequate backup processes leading to data loss.",
            "Lack of clarity in data classification practices."
        ],
        "calculated_risk_score": 66.67,
        "compliance_requirements": {
            "GDPR": [
                "Data encryption standards",
                "Data minimization and access controls"
            ],
            "HIPAA": [
                "Encryption of PHI",
                "Regular backup and disaster recovery"
            ],
            "ISO_27001": [
                "Data classification policies",
                "Secure data disposal procedures"
            ]
        }
    },
    "Incident_Response": {
        "domain_name": "Incident Response",
        "keywords": [
            "Incident Response Plan",
            "Forensics",
            "Threat Analysis",
            "Containment",
            "Eradication",
            "Recovery",
            "Post-Incident Review",
            "Communication Plan",
            "Incident Logging",
            "Playbooks"
        ],
        "preventive_measures": [
            "Develop and maintain a comprehensive incident response plan.",
            "Conduct regular drills and training exercises for the incident response team.",
            "Implement incident detection tools to identify threats early.",
            "Conduct post-incident reviews to analyze responses and improve processes."
        ],
        "followup_questions": [
            "How often is the incident response plan updated and tested?",
            "What tools do you use for incident detection and monitoring?",
            "Can you describe a recent incident and the steps taken to respond?",
            "How do you communicate with stakeholders during and after an incident?"
        ],
        "risk_indicators": [
            "Lack of documented incident response plan.",
            "Frequent delays in incident detection and response.",
            "Inconsistent training for incident response personnel.",
            "Inadequate communication strategies during incidents."
        ],
        "calculated_risk_score": 61.67,
        "compliance_requirements": {
            "NIST": [
                "Incident response planning",
                "Log analysis and continuous monitoring"
            ],
            "ISO_27035": [
                "Structured incident handling procedures",
                "Post-incident reviews"
            ],
            "CIS": [
                "Incident response training",
                "Effective communication protocols"
            ]
        }
    },
    "Compliance_and_Governance": {
        "domain_name": "Compliance and Governance",
        "keywords": [
            "Regulatory Compliance",
            "Risk Assessment",
            "Audit Trails",
            "Policy Enforcement",
            "Security Frameworks",
            "Data Protection Regulations",
            "Internal Controls",
            "Compliance Audits",
            "Governance Policies",
            "Incident Reporting"
        ],
        "preventive_measures": [
            "Ensure compliance with applicable regulations (e.g., GDPR, HIPAA).",
            "Regularly review and update security policies and procedures.",
            "Maintain detailed audit logs of all security-related activities.",
            "Conduct regular risk assessments to identify and mitigate compliance gaps."
        ],
        "followup_questions": [
            "What measures are in place to ensure ongoing compliance with regulations?",
            "How often are security policies reviewed and updated?",
            "Can you provide details on the last compliance audit and its findings?",
            "How is employee training on compliance and governance handled?"
        ],
        "risk_indicators": [
            "Non-compliance with regulatory requirements.",
            "Lack of up-to-date security policies.",
            "Inadequate audit trails for security events.",
            "Insufficient risk assessments leading"
        ],
        "calculated_risk_score": 68.33,
        "compliance_requirements": {
            "SOX": [
                "Internal controls and audit trails",
                "Regular risk assessments"
            ],
            "GDPR": [
                "Data protection measures",
                "Compliance reporting"
            ],
            "NIST": [
                "Governance frameworks",
                "Security policy development"
            ],
            "ISO_27001": [
                "Information security management systems (ISMS)",
                "Compliance audits"
            ]
        }
    }
}

# Save the dictionary to a JSON file with pretty formatting
with open("security_data.json", "w") as file:
    json.dump(data, file, indent=4)

print("JSON data saved to 'security_data.json'")
