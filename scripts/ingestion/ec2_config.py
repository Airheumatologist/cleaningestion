"""
EC2 Connection Configuration for PMC Data Pipeline.

Configure SSH connection details to access existing PMC data on EC2 instance.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


class EC2Config:
    """Configuration for EC2 SSH connection."""
    
    # SSH Connection Details
    EC2_HOST: Optional[str] = os.getenv("EC2_HOST")
    EC2_USER: str = os.getenv("EC2_USER", "ec2-user")
    EC2_SSH_KEY_PATH: Optional[str] = os.getenv("EC2_SSH_KEY_PATH")
    EC2_SSH_PORT: int = int(os.getenv("EC2_SSH_PORT", "22"))
    
    # Data Paths on EC2
    EC2_PMC_DATA_PATH: str = os.getenv("EC2_PMC_DATA_PATH", "/data/pmc_fulltext")
    EC2_PMC_XML_PATH: str = os.getenv("EC2_PMC_XML_PATH", "/data/pmc_fulltext/xml")
    EC2_PMC_JSONL_PATH: str = os.getenv("EC2_PMC_JSONL_PATH", "/data/pmc_fulltext/articles.jsonl")
    
    # AWS Configuration (optional, for AWS CLI access)
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_PROFILE: Optional[str] = os.getenv("AWS_PROFILE")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required EC2 configuration is set."""
        errors = []
        
        if not cls.EC2_HOST:
            errors.append("EC2_HOST not set in .env")
        if not cls.EC2_SSH_KEY_PATH:
            errors.append("EC2_SSH_KEY_PATH not set in .env")
        elif not Path(cls.EC2_SSH_KEY_PATH).exists():
            errors.append(f"SSH key file not found: {cls.EC2_SSH_KEY_PATH}")
        
        if errors:
            raise ValueError(
                "Missing EC2 configuration:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        
        return True
    
    @classmethod
    def get_ssh_command(cls, remote_command: str) -> str:
        """Generate SSH command to execute on EC2."""
        return (
            f"ssh -i {cls.EC2_SSH_KEY_PATH} "
            f"-p {cls.EC2_SSH_PORT} "
            f"-o StrictHostKeyChecking=no "
            f"{cls.EC2_USER}@{cls.EC2_HOST} "
            f"'{remote_command}'"
        )
    
    @classmethod
    def get_scp_command(cls, local_path: str, remote_path: str, direction: str = "to") -> str:
        """Generate SCP command to transfer files.
        
        Args:
            local_path: Local file path
            remote_path: Remote file path on EC2
            direction: "to" (local -> remote) or "from" (remote -> local)
        """
        remote = f"{cls.EC2_USER}@{cls.EC2_HOST}:{remote_path}"
        
        if direction == "to":
            return (
                f"scp -i {cls.EC2_SSH_KEY_PATH} "
                f"-P {cls.EC2_SSH_PORT} "
                f"-o StrictHostKeyChecking=no "
                f"{local_path} {remote}"
            )
        else:  # from
            return (
                f"scp -i {cls.EC2_SSH_KEY_PATH} "
                f"-P {cls.EC2_SSH_PORT} "
                f"-o StrictHostKeyChecking=no "
                f"{remote} {local_path}"
            )


if __name__ == "__main__":
    print("🔧 EC2 Configuration Check")
    print("=" * 50)
    try:
        EC2Config.validate()
        print(f"✅ EC2_HOST: {EC2Config.EC2_HOST}")
        print(f"✅ EC2_USER: {EC2Config.EC2_USER}")
        print(f"✅ EC2_SSH_KEY_PATH: {EC2Config.EC2_SSH_KEY_PATH}")
        print(f"✅ EC2_PMC_DATA_PATH: {EC2Config.EC2_PMC_DATA_PATH}")
        print(f"✅ EC2_PMC_XML_PATH: {EC2Config.EC2_PMC_XML_PATH}")
        print("\n✅ EC2 configuration validated!")
    except ValueError as e:
        print(f"❌ Configuration error:\n{e}")

