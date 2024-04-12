use clap::Parser;

#[derive(Debug, Parser)]
struct Args { 
    /// The Sui package id associated with the Atoma call contract
    #[arg(long)]
    pub package_id: String
}

[tokio::main]
fn main() {
    let args = Args::parse();
    let package_id = args.package_id;
    
}