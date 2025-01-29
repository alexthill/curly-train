use std::io::{self, Cursor};
use std::path::{Path, PathBuf};

pub fn load<P: AsRef<Path>>(path: P) -> Result<Cursor<Vec<u8>>, io::Error> {
    use std::fs::File;
    use std::io::Read;

    let mut buf = Vec::new();
    let mut file = File::open(path)?;
    file.read_to_end(&mut buf)?;
    Ok(Cursor::new(buf))
}

#[derive(Debug, Default, Clone)]
pub struct Carousel {
    dir: &'static str,
    curr: usize,
}

impl Carousel {
    pub fn new(dir: &'static str) -> Self {
        Self { dir, curr: 0 }
    }

    pub fn set_dir(&mut self, dir: &'static str) {
        self.dir = dir;
    }

    pub fn get_next<F>(&mut self, offset: isize, filter: F) -> Result<PathBuf, io::Error>
    where
        F: Fn(&Path) -> bool,
    {
        let mut files = std::fs::read_dir(self.dir)?
            .filter_map(|path| {
                let path = path.ok()?;
                if !path.file_type().ok()?.is_file() {
                    return None;
                }
                let path = path.path();
                if !filter(&path) {
                    return None;
                }
                Some(path)
            })
            .collect::<Vec<_>>();
        if files.is_empty() {
            return Err(io::Error::new(io::ErrorKind::Other, "no matching file found"));
        }
        files.sort();
        // take euclidian remainder and not modulus to get a positive value
        self.curr = (self.curr as isize + offset).rem_euclid(files.len() as isize) as usize;
        Ok(files[self.curr].clone())
    }
}
