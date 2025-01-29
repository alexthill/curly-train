use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead};
use std::num::NonZeroU32;
use std::str;

#[derive(Debug, Default, Clone)]
pub struct Obj {
    pub vertices: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub faces: Vec<([Indices; 3], Option<Indices>)>,
}

impl Obj {
    pub fn from_reader(reader: impl BufRead) -> Result<Self, (ObjError, usize)> {
        let mut obj = Self::default();
        for (line_num, line) in reader.split(b'\n').enumerate() {
            if let Err(err) = obj.parse_line(line) {
                return Err((err, line_num + 1));
            }
        }
        Ok(obj)
    }

    fn parse_line(&mut self, line: Result<Vec<u8>, io::Error>) -> Result<(), ObjError> {
        let line = line?;
        if line.is_empty() || line[0] == b'#' {
            return Ok(());
        }

        let mut parts = line.split(|c| c.is_ascii_whitespace())
            .filter(|part| !part.is_empty());
        let Some(iden) = parts.next() else { return Ok(()) };
        match iden {
            b"f" => self.faces.push((
                [
                    Self::parse_part::<_, 3>(0, parts.next())?,
                    Self::parse_part::<_, 3>(1, parts.next())?,
                    Self::parse_part::<_, 3>(2, parts.next())?,
                ],
                parts.next().map(|part| Self::parse_part::<_, 3>(3, Some(part))).transpose()?,
            )),
            b"v" => self.vertices.push([
                Self::parse_part::<_, 3>(0, parts.next())?,
                Self::parse_part::<_, 3>(1, parts.next())?,
                Self::parse_part::<_, 3>(2, parts.next())?,
            ]),
            b"vt" => self.tex_coords.push([
                Self::parse_part::<_, 2>(0, parts.next())?,
                Self::parse_part::<_, 2>(1, parts.next())?,
            ]),
            // not implemented
            b"g" | b"o" | b"s" | b"vn" | b"mtllib" | b"usemtl" => return Ok(()),
            other => {
                return Err(ObjError::InvalidIden(String::from_utf8_lossy(other).into_owned()));
            }
        };
        if let Some(next) = parts.next() {
            if next[0] != b'#' {
                return Err(ObjError::TooManyNums);
            }
        }
        Ok(())
    }

    pub fn normalize(&self) -> Result<NormalizedObj, ObjError> {
        let mut map = HashMap::<Indices, u32>::new();
        let mut nobj = NormalizedObj::default();
        for face in self.faces.iter() {
            fn map_indices(
                indices: Indices,
                obj: &Obj,
                nobj: &mut NormalizedObj,
                map: &mut HashMap<Indices, u32>,
            ) -> Result<u32, ObjError> {
                let vert_idx = *map.entry(indices).or_insert(nobj.vertices.len() as u32);
                if vert_idx == nobj.vertices.len() as u32 {
                    let pos_coords = *obj.vertices.get(indices.vertex.get() as usize - 1)
                        .ok_or(ObjError::InvalidVertexIndex(indices.vertex.into()))?;
                    let tex_coords = if let Some(tex_coords_idx) = indices.texture {
                        nobj.has_tex_coords = true;
                        *obj.tex_coords.get(tex_coords_idx.get() as usize - 1)
                            .ok_or(ObjError::InvalidTextureIndex(tex_coords_idx.into()))?
                    } else {
                        [0.; 2]
                    };
                    nobj.vertices.push(Vertex { pos_coords, tex_coords });
                }
                Ok(vert_idx)
            }

            let indices: Vec<_> = if let Some(v4) = face.1 {
                let v = face.0;
                [v[0], v[1], v[2], v[2], v4, v[0]]
                    .map(|x| map_indices(x, self, &mut nobj, &mut map))
                    .into_iter().collect::<Result<_, _>>()?
            } else {
                face.0
                    .map(|x| map_indices(x, self, &mut nobj, &mut map))
                    .into_iter().collect::<Result<_, _>>()?
            };
            nobj.indices.extend(indices);
        }
        Ok(nobj)
    }

    fn parse_part<T, const N: u32>(n: u32, part: Option<&[u8]>) -> Result<T, ObjError>
    where
        T: str::FromStr,
    {
        match part {
            Some(part) => str::from_utf8(part)
                .map_err(|_| ObjError::InvalidNum(String::from_utf8_lossy(part).into_owned()))?
                .parse()
                .map_err(|_| ObjError::InvalidNum(String::from_utf8_lossy(part).into_owned())),
            None => Err(ObjError::NotEnoughNums(n, N)),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct NormalizedObj {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub has_tex_coords: bool,
}

impl NormalizedObj {
    pub fn from_reader(reader: impl BufRead) -> Result<Self, ObjError> {
        Obj::from_reader(reader).map_err(|(err, _)| err)?.normalize()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub pos_coords: [f32; 3],
    pub tex_coords: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Indices {
    pub vertex: NonZeroU32,
    pub texture: Option<NonZeroU32>,
    pub normal: Option<NonZeroU32>,
}

impl str::FromStr for Indices {
    type Err = ObjError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split('/');
        let Some(part) = parts.next() else {
            return Err(ObjError::NotEnoughNums(0, 1));
        };
        let vertex = part.parse().map_err(|_| ObjError::InvalidNum(part.to_owned()))?;
        let texture = match parts.next() {
            Some(part) if !part.is_empty() =>
                Some(part.parse().map_err(|_| ObjError::InvalidNum(part.to_owned()))?),
            _ => None,
        };
        let normal = if let Some(part) = parts.next() {
            Some(part.parse().map_err(|_| ObjError::InvalidNum(part.to_owned()))?)
        } else {
            None
        };

        Ok(Self { vertex, texture, normal })
    }
}

#[derive(Debug)]
pub enum ObjError {
   InvalidIden(String),
   InvalidNum(String),
   InvalidTextureIndex(u32),
   InvalidVertexIndex(u32),
   Io(io::Error),
   NotEnoughNums(u32, u32),
   TooManyNums,
}

impl fmt::Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIden(iden) => write!(f, "Invalid identifier at line start: {iden}"),
            Self::InvalidNum(num) => write!(f, "Invalid number: {num}"),
            Self::InvalidTextureIndex(idx) => write!(f, "Invalid texture index: {idx}"),
            Self::InvalidVertexIndex(idx) => write!(f, "Invalid vertex index: {idx}"),
            Self::Io(err) => write!(f, "IO error: {err}"),
            Self::NotEnoughNums(found, expt) =>
                write!(f, "Not enough numbers at line: found {found} expected at least {expt}"),
            Self::TooManyNums => write!(f, "Too many numbers at line"),
        }
    }
}

impl Error for ObjError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for ObjError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::{BufReader, Cursor};
    use std::path::Path;

    #[test]
    fn parse_vertice() {
        let file = "v 1 2.2  3.14159";
        let obj = Obj::from_reader(Cursor::new(file.as_bytes())).expect("failed to parse");
        assert_eq!(obj.vertices, [[1., 2.2, 3.14159]]);
    }

    #[test]
    fn parse_vertices() {
        let file = "v 1 2.2  3.14159\nv 1 2 3   ";
        let obj = Obj::from_reader(Cursor::new(file.as_bytes())).expect("failed to parse");
        assert_eq!(obj.vertices, [[1., 2.2, 3.14159], [1., 2., 3.]]);
    }

    #[test]
    fn parse_obj_file_chalet() {
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets").join("models");
        let file = File::open(src_dir.join("chalet.obj")).unwrap();
        let reader = BufReader::new(file);
        let obj = Obj::from_reader(reader).expect("failed to parse");
        assert_eq!(obj.vertices.len(), 234246);
        assert_eq!(obj.tex_coords.len(), 265645);
        assert_eq!(obj.faces.len(), 500000);

        let nobj = obj.normalize().expect("failed to normalize");
        assert_eq!(nobj.vertices.len(), 265645);
        assert_eq!(nobj.indices.len(), 500000 * 3);
    }

    #[test]
    fn parse_obj_file_42() {
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets").join("models");
        let file = File::open(src_dir.join("42.obj")).unwrap();
        let reader = BufReader::new(file);
        let obj = Obj::from_reader(reader).expect("failed to parse");
        assert_eq!(obj.vertices.len(), 42);
        assert_eq!(obj.tex_coords.len(), 0);
        assert_eq!(obj.faces.len(), 47);

        let nobj = obj.normalize().expect("failed to normalize");
        assert_eq!(nobj.vertices.len(), 42);
        assert_eq!(nobj.indices.len(), 47 * 3 + 29 * 3);
    }

    #[test]
    fn parse_normalize() {
        let file = r#"
v 1.1 1.2 1.3
v 2.1 2.2 2.3
v 3.1 3.2 3.3
vt 0.1 0.2
vt 0.3 0.4
vt 0.5 0.6
f 1/1 2/2 3/3
"#;
        let obj = Obj::from_reader(Cursor::new(file.as_bytes())).expect("failed to parse");
        assert_eq!(obj.vertices, [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]]);
        assert_eq!(obj.tex_coords, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);

        let nobj = obj.normalize().expect("failed to normalize");
        assert_eq!(nobj.vertices, [
            Vertex { pos_coords: [1.1, 1.2, 1.3], tex_coords: [0.1, 0.2] },
            Vertex { pos_coords: [2.1, 2.2, 2.3], tex_coords: [0.3, 0.4] },
            Vertex { pos_coords: [3.1, 3.2, 3.3], tex_coords: [0.5, 0.6] },
        ]);
        assert_eq!(nobj.indices, [0, 1, 2]);
    }

    #[test]
    fn parse_normalize_complex() {
        let file = r#"
v 1.1 1.2 1.3
v 2.1 2.2 2.3
v 3.1 3.2 3.3
vt 0.1 0.2
vt 0.3 0.4
vt 0.5 0.6
vt 0.7 0.8
f 1/1 2/2 3/3
f 2/1 1/2 3/4
"#;
        let obj = Obj::from_reader(Cursor::new(file.as_bytes())).expect("failed to parse");
        assert_eq!(obj.vertices, [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]]);
        assert_eq!(obj.tex_coords, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]);

        let nobj = obj.normalize().expect("failed to normalize");
        assert_eq!(nobj.vertices, [
            Vertex { pos_coords: [1.1, 1.2, 1.3], tex_coords: [0.1, 0.2] },
            Vertex { pos_coords: [2.1, 2.2, 2.3], tex_coords: [0.3, 0.4] },
            Vertex { pos_coords: [3.1, 3.2, 3.3], tex_coords: [0.5, 0.6] },
            Vertex { pos_coords: [2.1, 2.2, 2.3], tex_coords: [0.1, 0.2] },
            Vertex { pos_coords: [1.1, 1.2, 1.3], tex_coords: [0.3, 0.4] },
            Vertex { pos_coords: [3.1, 3.2, 3.3], tex_coords: [0.7, 0.8] },
        ]);
        assert_eq!(nobj.indices, [0, 1, 2, 3, 4, 5]);
    }
}
